#pragma once
#include <vector>
#include <ml/include/ml.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <atomic>

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
  std::string PrintExpDetail() const;

private:
  int num_samples_;
  int num_features_;    // # features across all workers.
  bool sample_one_based_;
  bool snappy_compressed_;

  // Columns of design matrix. Each thread is in charge of a few columns.
  std::vector<petuum::ml::SparseFeature<float>*> X_cols_;
  petuum::ml::DenseFeature<float> Y_;

  std::atomic<int32_t> thread_counter_;
};

}  // namespace lasso
