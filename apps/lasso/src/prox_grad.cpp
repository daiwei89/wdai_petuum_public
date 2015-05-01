#include "prox_grad.hpp"
#include "common.hpp"
#include <vector>
#include <ml/include/ml.hpp>

namespace lasso {

ProxGrad::ProxGrad(const ProxGradConfig& config) :
  worker_rank_(config.worker_rank),
  num_workers_(FLAGS_num_clients * FLAGS_num_threads),
  num_samples_(config.num_samples),
  feature_start_(config.feature_start),
  feature_end_(config.feature_end),
  num_features_(feature_end_ - feature_start_),
  beta_(num_features_),
  r_(num_samples_),
  w_table_(config.w_table) {
    CHECK_GT(num_features_, 0);
  }

void ProxGrad::ProxStep(
    const std::vector<petuum::ml::SparseFeature<float>*>& X_cols,
    const petuum::ml::DenseFeature<float>& w_other,
    const petuum::ml::DenseFeature<float>& y,
    float lr) {
  CHECK_EQ(num_samples_, w_other.GetFeatureDim());
  CHECK_EQ(num_samples_, y.GetFeatureDim());
  // w = r_ + w_other
  petuum::ml::DenseFeature<float> w = r_;
  //LOG(INFO) << "w_other: " << w_other.ToString();
  //LOG(INFO) << "y: " << y.ToString();
  //LOG(INFO) << "w_me = r_: " << w.ToString();
  FeatureScaleAndAdd(1, w_other, &w);
  //LOG(INFO) << "w + w_other: " << w.ToString();

  // w -= y
  FeatureScaleAndAdd(-1, y, &w);
  //LOG(INFO) << "w - y: " << w.ToString();

  petuum::ml::DenseFeature<float> grad(num_features_);
  for (int j = 0; j < num_features_; ++j) {
    grad[j] = SparseDenseFeatureDotProduct(
        *(X_cols[feature_start_ + j]), w);
    CHECK_EQ(feature_start_, 0);
    //LOG(INFO) << "X[" << j << "]: " << X_cols[j]->ToString();
  }
  //LOG(INFO) << "grad: " << grad.ToString();

  // delta_ = prox[beta - learning_rate * X_i^T (w - y)] - beta_
  petuum::ml::DenseFeature<float> delta_ = beta_;
  //LOG(INFO) << "beta: " << beta_.ToString();
  FeatureScaleAndAdd(-lr, grad, &delta_);
  //LOG(INFO) << "beta - lr * grad: " << delta_.ToString();
  SoftThreshold(lr * FLAGS_lambda, &delta_);
  //LOG(INFO) << "prox(beta - lr * grad): " << delta_.ToString();
  //LOG(INFO) << "lr * lambda: " << lr * FLAGS_lambda;
  FeatureScaleAndAdd(-1, beta_, &delta_);
  //LOG(INFO) << "prox(beta - grad) - beta: " << delta_.ToString();

  // beta_ += delta_
  FeatureScaleAndAdd(1, delta_, &beta_);
  //LOG(INFO) << "new beta: " << beta_.ToString();

  // r_delta = X_i * \Delta_i
  petuum::ml::DenseFeature<float> r_delta(num_samples_);
  for (int j = 0; j < num_features_; ++j) {
    FeatureScaleAndAdd(delta_[j],
        *(X_cols[feature_start_ + j]), &r_delta);
  }
  //LOG(INFO) << "update: " << r_delta.ToString();
  petuum::UpdateBatch<float> update(num_samples_);
  for (int i = 0; i < num_samples_; ++i) {
    update.UpdateSet(i, i, r_delta[i]);
    r_[i] += r_delta[i];    // r_ = r_ + X_i * \Delta
  }
  w_table_.BatchInc(worker_rank_, update);
  w_table_.BatchInc(num_workers_, update);
}

// Sum of sqloss for all data.
float ProxGrad::EvalSqLoss(
    const petuum::ml::DenseFeature<float>& w_all,
    const petuum::ml::DenseFeature<float>& y) const {
  petuum::ml::DenseFeature<float> sq_err = w_all;
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
