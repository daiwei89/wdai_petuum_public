#pragma once
#include <vector>
#include <ml/include/ml.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <atomic>
#include "prox_grad.hpp"

namespace lasso {

class LassoEngine {
public:
  LassoEngine();
  ~LassoEngine();

  // Return # of samples (# of columns if w_table).
  int ReadData();

  // Can be called concurrently.
  void Start();

private:
  std::string PrintExpDetail(const ProxGrad& pg) const;

private:
  int num_samples_{0};
  int num_features_{0};    // # features across all workers.
  bool sample_one_based_{false};
  bool snappy_compressed_{false};

  // Partition this worker handles.
  int par_begin_;
  int par_end_;

  // Columns of design matrix. Each thread is in charge of a few columns.
  std::vector<petuum::ml::SparseFeature<float>*> X_cols_;
  petuum::ml::DenseFeature<float> Y_;

  std::atomic<int32_t> thread_counter_;
};

}  // namespace lasso
