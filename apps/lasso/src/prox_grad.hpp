#pragma once

#include <vector>
#include <ml/include/ml.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace lasso {

struct ProxGradConfig {
  int worker_rank = -1;
  int feature_start = 0;
  int feature_end = 0;
  int num_samples = 0;
  petuum::Table<float> w_table;
};

class ProxGrad {
public:
  ProxGrad(const ProxGradConfig& config);

  void ProxStep(
      const std::vector<petuum::ml::SparseFeature<float>*>& X_cols,
      const petuum::ml::DenseFeature<float>& y,
      float lr) ;

  float EvalSqLoss(
      const petuum::ml::DenseFeature<float>& y);

  float EvalL1Penalty() const;

  // Cound # of non-zeros in beta (sharded to workers).
  int EvalBetaNNZ() const;

private:
  void SoftThreshold(float threshold,
      petuum::ml::DenseFeature<float>* x);

private:
  int worker_rank_;
  int num_workers_;
  int num_samples_;
  int feature_start_;
  int feature_end_;
  int num_features_;    // # features in this worker.
  petuum::ml::DenseFeature<float> beta_;    // [num_features_ x 1]
  petuum::ml::DenseFeature<float> r_;       // [num_samples_ x 1]
  //petuum::ml::DenseFeature<float> delta_;   // [num_features_ x 1]
  petuum::Table<float> w_table_;
};

}  // namespace lasso
