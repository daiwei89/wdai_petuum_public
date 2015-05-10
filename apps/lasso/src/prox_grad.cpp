#include "prox_grad.hpp"
#include "common.hpp"
#include <cmath>
#include <vector>
#include <ml/include/ml.hpp>
#include <cstdint>

namespace lasso {

ProxGrad::ProxGrad(const ProxGradConfig& config) :
  worker_rank_(config.worker_rank),
  num_workers_(FLAGS_num_clients * FLAGS_num_threads),
  num_samples_(config.num_samples),
  feature_start_(config.feature_start),
  feature_end_(config.feature_end),
  num_features_(feature_end_ - feature_start_),
  beta_(num_features_),
  r_(num_samples_) {
    w_table_ =
      petuum::PSTableGroup::GetTableOrDie<float>(FLAGS_w_table_id);
    unused_table_ =
      petuum::PSTableGroup::GetTableOrDie<float>(FLAGS_unused_table_id);
    staleness_table_ =
      petuum::PSTableGroup::GetTableOrDie<int64_t>(FLAGS_staleness_table_id);
    CHECK_GT(num_features_, 0);
  }

void ProxGrad::ProxStep(
    const std::vector<petuum::ml::SparseFeature<float>*>& X_cols,
    const petuum::ml::DenseFeature<float>& y,
    float lr, int my_clock) {
  petuum::RowAccessor row_acc;
  // Read rows but not really use them
  for (int z = 0; z < FLAGS_num_unused_rows; ++z) {
    unused_table_.Get(z, &row_acc);
  }
  w_table_.Get(0, &row_acc);
  const auto& w_all_row = row_acc.Get<petuum::DenseRow<float>>();
  std::vector<float> row_vec;
  w_all_row.CopyToVector(&row_vec);

  // Record clock differences. staleness_dist[0] corresponds to
  // -FLAGS_staleness.
  std::vector<int> staleness_dist(2 * FLAGS_staleness + 1);
  for (int i = 0; i < num_workers_; ++i) {
    if (i == worker_rank_) {
      continue;   // skip my own clock.
    }
    int clock_diff = my_clock - row_vec[i + num_samples_];
    CHECK_LE(std::abs(clock_diff), FLAGS_staleness) << "my_clock: " << my_clock
      << " other clock: " << row_vec[i + num_samples_];
    staleness_dist[clock_diff + FLAGS_staleness]++;
  }
  petuum::UpdateBatch<int64_t> staleness_update(2 * FLAGS_staleness + 1);
  for (int s = 0; s < 2*FLAGS_staleness + 1; ++s) {
    staleness_update.UpdateSet(s, s, staleness_dist[s]);
  }
  staleness_table_.BatchInc(0, staleness_update);

  std::vector<float> w_only_vec(row_vec.begin(),
      row_vec.begin() + num_samples_);
  petuum::ml::DenseFeature<float> w_all(w_only_vec);
  CHECK_EQ(num_samples_, w_all.GetFeatureDim());
  CHECK_EQ(num_samples_, y.GetFeatureDim());
  // w -= y
  FeatureScaleAndAdd(-1, y, &w_all);

  // gradient = 2 * X_k^T (w_all - y)
  petuum::ml::DenseFeature<float> grad(num_features_);
  for (int j = 0; j < num_features_; ++j) {
    grad[j] = 2 * SparseDenseFeatureDotProduct(
        *(X_cols[feature_start_ + j]), w_all);
  }

  petuum::ml::DenseFeature<float> new_beta = beta_;
  FeatureScaleAndAdd(-lr, grad, &new_beta);
  SoftThreshold(lr * FLAGS_lambda, &new_beta);

  // delta_k^{c+1} = beta_k^{c+1} - beta_k^c
  petuum::ml::DenseFeature<float> delta = new_beta;
  FeatureScaleAndAdd(-1, beta_, &delta);
  beta_ = new_beta;

  // X_k * delta_k^{c+1}
  petuum::ml::DenseFeature<float> X_delta(num_samples_);
  for (int j = 0; j < num_features_; ++j) {
    FeatureScaleAndAdd(delta[j],
        *(X_cols[feature_start_ + j]), &X_delta);
  }

  petuum::UpdateBatch<float> update(num_samples_ + 1);
  for (int i = 0; i < num_samples_; ++i) {
    update.UpdateSet(i, i, X_delta[i]);
  }
  // Increment clock by 1.
  update.UpdateSet(num_samples_, worker_rank_ + num_samples_, 1);
  w_table_.BatchInc(0, update);

  // Update unused rows
  petuum::UpdateBatch<float> unused_update(FLAGS_num_unused_cols);
  for (int i = 0; i < FLAGS_num_unused_cols; ++i) {
    unused_update.UpdateSet(i, i, 1);
  }
  for (int z = 0; z < FLAGS_num_unused_rows; ++z) {
    unused_table_.BatchInc(z, unused_update);
  }
}

// Sum of sqloss for all data.
float ProxGrad::EvalSqLoss(
    const petuum::ml::DenseFeature<float>& y) {
  petuum::ml::DenseFeature<float> w_all;
  petuum::RowAccessor row_acc;
  w_table_.Get(0, &row_acc);
  const auto& w_all_row = row_acc.Get<petuum::DenseRow<float>>();
  std::vector<float> row_vec;
  w_all_row.CopyToVector(&row_vec);
  std::vector<float> w_only_vec(row_vec.begin(),
      row_vec.begin() + num_samples_);
  //w_all_row.CopyToDenseFeature(&w_all);
  //petuum::ml::DenseFeature<float> sq_err = w_all;
  petuum::ml::DenseFeature<float> sq_err(w_only_vec);
  FeatureScaleAndAdd(-1, y, &sq_err);
  float sq_loss = 0.;
  for (int j = 0; j < num_samples_; ++j) {
    sq_loss += sq_err[j] * sq_err[j];
  }
  return 0.5 * sq_loss;
}

// L1 penalty from the local beta segment.
float ProxGrad::EvalL1Penalty() const {
  float L1_penalty = 0.;
  for (int j = 0; j < beta_.GetFeatureDim(); ++j) {
    L1_penalty += (beta_[j] > 0) ? beta_[j] : -beta_[j];
  }
  return FLAGS_lambda * L1_penalty;
}

int ProxGrad::EvalBetaNNZ() const {
  int nnz = 0;
  for (int j = 0; j < num_features_; ++j) {
    nnz += (beta_[j] != 0) ? 1 : 0;
  }
  return nnz;
}

void ProxGrad:: SoftThreshold(float threshold,
    petuum::ml::DenseFeature<float>* x) {
  for (int i = 0; i < x->GetFeatureDim(); ++i) {
    if ((*x)[i] > threshold) {
      (*x)[i] -= threshold;
    } else if ((*x)[i] < -threshold) {
      (*x)[i] += threshold;
    } else {
      (*x)[i] = 0.;
    }
  }
}

}  // namespace lasso
