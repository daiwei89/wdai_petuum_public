#pragma once

#include <vector>
#include <ml/include/ml.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <random>

namespace lasso {

struct ProxGradConfig {
  int worker_rank = -1;
  int feature_start = 0;
  int feature_end = 0;
  int num_samples = 0;
  int num_reps = 1;
};

class ProxGrad {
public:
  ProxGrad(const ProxGradConfig& config);

  void ProxStep(
      const std::vector<petuum::ml::SparseFeature<float>*>& X_cols,
      const petuum::ml::DenseFeature<float>& y,
      float lr, int my_clock);

  float EvalSqLoss(
      const petuum::ml::DenseFeature<float>& y);

  float EvalL1Penalty() const;

  // Cound # of non-zeros in beta (sharded to workers).
  int EvalBetaNNZ() const;

  int GetSampleSize() const {
    return num_feature_samples_;
  }

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
  // stochastic version uses only this many coordinates in each step
  int num_feature_samples_;
  int num_reps_;
  petuum::ml::DenseFeature<float> beta_;    // [num_features_ x 1]
  petuum::ml::DenseFeature<float> r_;       // [num_samples_ x 1]
  //petuum::ml::DenseFeature<float> delta_;   // [num_features_ x 1]
  petuum::Table<float> w_table_;
  petuum::Table<float> unused_table_;
  petuum::Table<int64_t> staleness_table_;
  std::random_device r;
  std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937 rand_eng{seed2};
  std::uniform_real_distribution<float> uniform_dist{0, 1};
};

}  // namespace lasso
