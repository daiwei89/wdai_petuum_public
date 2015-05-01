#include "lasso_engine.hpp"
#include "loss_recorder.hpp"
#include "common.hpp"
#include "prox_grad.hpp"
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <ml/include/ml.hpp>
#include <fstream>
#include <string>
#include <sstream>


namespace lasso {

LassoEngine::LassoEngine() : thread_counter_(0) {
  // Apepnd client_id if the train_data isn't global.
  std::string meta_file = FLAGS_X_file
    + (FLAGS_global_data ? "" : "." + std::to_string(FLAGS_client_id))
    + ".meta";
  petuum::ml::MetafileReader mreader(meta_file);
  // num features in the X_file (# of rows)
  num_features_ = mreader.get_int32("num_features");
  // num_samples_ is the dim of each feature column.
  num_samples_ = mreader.get_int32("num_samples");
  Y_ = petuum::ml::DenseFeature<float>(num_samples_);
  CHECK_EQ("libsvm", mreader.get_string("format")) << "Only support libsvm";
  sample_one_based_ = mreader.get_bool("sample_one_based");
  snappy_compressed_ = mreader.get_bool("snappy_compressed");
}

LassoEngine::~LassoEngine() {
  for (auto& row_ptr : X_cols_) {
    delete row_ptr;
  }
}

int LassoEngine::ReadData() {
  //std::string X_file = FLAGS_X_file
  //  + (FLAGS_global_data ? "" : "." + std::to_string(FLAGS_client_id));
  std::string X_file = FLAGS_X_file;
  LOG(INFO) << "Reading X_file: " << FLAGS_X_file;
  // Note feature and sample are transposed (each row is a feature)
  std::vector<int> feature_idx;
  petuum::ml::ReadDataLabelLibSVM(X_file, num_samples_, num_features_,
      reinterpret_cast<std::vector<
      petuum::ml::AbstractFeature<float>*>*>(&X_cols_),
      &feature_idx, sample_one_based_, false,  // label is real value
      snappy_compressed_);

  // Read y
  //std::string Y_file = FLAGS_Y_file
  //  + (FLAGS_global_data ? "" : "." + std::to_string(FLAGS_client_id));
  std::string Y_file = FLAGS_Y_file;
  std::ifstream y_stream(Y_file);
  CHECK(y_stream);
  std::string line;
  int i = 0;
  while (getline(y_stream, line)) {
    CHECK_LT(i, num_samples_);
    Y_[i++] = std::stof(line);
  }
  LOG(INFO) << "Read data: num_features: " << num_features_
    << " num_samples: " << num_samples_;
  return num_samples_;
}

void LassoEngine::Start() {
  petuum::PSTableGroup::RegisterThread();

  // All threads must register in the same order.
  petuum::LossRecorder loss_recorder;
  loss_recorder.RegisterField("Epoch");
  loss_recorder.RegisterField("Time");
  loss_recorder.RegisterField("FullLoss");
  loss_recorder.RegisterField("BetaNNZ");   // nnz in regressor beta

  // Initialize local thread data structures.
  int thread_id = thread_counter_++;

  petuum::HighResolutionTimer total_timer;

  int worker_rank = FLAGS_client_id * FLAGS_num_threads + thread_id;
  int num_workers = FLAGS_num_clients * FLAGS_num_threads;
  int num_features_per_thread = num_features_ /
    FLAGS_num_clients / FLAGS_num_threads;
  int feature_start = worker_rank * num_features_per_thread;
  int feature_end = (worker_rank == FLAGS_num_clients * FLAGS_num_threads - 1)
    ? num_features_ : feature_start + num_features_per_thread;

  petuum::Table<float> w_table =
    petuum::PSTableGroup::GetTableOrDie<float>(FLAGS_w_table_id);
  ProxGradConfig pg_config;
  pg_config.worker_rank = worker_rank;
  pg_config.feature_start = feature_start;
  pg_config.feature_end = feature_end;
  pg_config.w_table = w_table;
  pg_config.num_samples = num_samples_;
  ProxGrad pg(pg_config);

  petuum::ml::DenseFeature<float> w_me;
  petuum::ml::DenseFeature<float> w_all;
  int eval_counter = 0;
  for (int epoch = 1; epoch <= FLAGS_num_epochs; ++epoch) {
    LOG(INFO) << "epoch: " << epoch;
    // w_other = w_all - w_me
    petuum::RowAccessor row_acc;
    w_table.Get(worker_rank, &row_acc);
    const auto& r = row_acc.Get<petuum::DenseRow<float>>();
    r.CopyToDenseFeature(&w_me);
    w_table.Get(num_workers, &row_acc);
    const auto& r_all = row_acc.Get<petuum::DenseRow<float>>();
    r_all.CopyToDenseFeature(&w_all);
    FeatureScaleAndAdd(-1, w_me, &w_all);   // w_all = w_other.
    pg.ProxStep(X_cols_, w_all, Y_, FLAGS_learning_rate);

    // Evaluate objective value.
    loss_recorder.IncLoss(eval_counter, "FullLoss", pg.EvalL1Penalty());
    loss_recorder.IncLoss(eval_counter, "BetaNNZ", pg.EvalBetaNNZ());
    if (worker_rank == 0) {
      LOG(INFO) << "recording epoch " << epoch;
      loss_recorder.IncLoss(eval_counter, "FullLoss", pg.EvalSqLoss(w_all, Y_));
      loss_recorder.IncLoss(eval_counter, "Epoch", epoch);
      loss_recorder.IncLoss(eval_counter, "Time", total_timer.elapsed());
    }
    if (worker_rank == 0 && epoch % 10 == 0) {
      LOG(INFO) << "Epoch " << epoch << " finished. Time: "
        << total_timer.elapsed() << " "
        << loss_recorder.PrintOneLoss(eval_counter - 1);
    }
    eval_counter++;
  }
  petuum::PSTableGroup::GlobalBarrier();

  if (worker_rank == 0) {
    LOG(INFO) << "\n" << PrintExpDetail() << loss_recorder.PrintAllLoss();
  }
  petuum::PSTableGroup::DeregisterThread();
}

std::string LassoEngine::PrintExpDetail() const {
  std::stringstream ss;
  ss << "num_clients: " << FLAGS_num_clients << std::endl;
  ss << "num_app_threads: " << FLAGS_num_threads << std::endl;
  ss << "staleness: " << FLAGS_staleness << std::endl;
  ss << "num_features: " << num_features_ << std::endl;
  ss << "lambda: " << FLAGS_lambda << std::endl;
  ss << "num_epochs: " << FLAGS_num_epochs << std::endl;
  ss << "learning_rate: " << FLAGS_learning_rate << std::endl;
  return ss.str();
}

}  // namespace lasso
