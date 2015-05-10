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
#include <cstdint>


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
  int num_feature_per_worker = num_features_ /
    FLAGS_num_clients / FLAGS_num_threads;
  int feature_start = worker_rank * num_feature_per_worker;
  int feature_end = (worker_rank == FLAGS_num_clients * FLAGS_num_threads - 1)
    ? num_features_ : feature_start + num_feature_per_worker;

  ProxGradConfig pg_config;
  pg_config.worker_rank = worker_rank;
  pg_config.feature_start = feature_start;
  pg_config.feature_end = feature_end;
  pg_config.num_samples = num_samples_;
  ProxGrad pg(pg_config);

  int eval_counter = 0;
  for (int epoch = 1; epoch <= FLAGS_num_epochs; ++epoch) {
    pg.ProxStep(X_cols_, Y_, FLAGS_learning_rate, epoch - 1);

    // Evaluate objective value.
    if (epoch == 1 || epoch % FLAGS_num_epochs_per_eval == 0) {
      loss_recorder.IncLoss(eval_counter, "FullLoss", pg.EvalL1Penalty());
      loss_recorder.IncLoss(eval_counter, "BetaNNZ", pg.EvalBetaNNZ());
      if (worker_rank == 0) {
        loss_recorder.IncLoss(eval_counter, "FullLoss", pg.EvalSqLoss(Y_));
        loss_recorder.IncLoss(eval_counter, "Epoch", epoch);
        loss_recorder.IncLoss(eval_counter, "Time", total_timer.elapsed());
        LOG(INFO) << "Epoch " << epoch << " finished. Time: "
          << total_timer.elapsed();
        if (eval_counter > 0) {
          LOG(INFO) << loss_recorder.PrintOneLoss(eval_counter - 1);
        }
      }
      eval_counter++;
    }
    petuum::PSTableGroup::Clock();
  }
  petuum::PSTableGroup::GlobalBarrier();

  if (worker_rank == 0) {
    LOG(INFO) << "\n" << PrintExpDetail() << loss_recorder.PrintAllLoss();
    std::string output_file = FLAGS_output_dir + "/loss";
    std::ofstream out(output_file);
    out << PrintExpDetail() << loss_recorder.PrintAllLoss();
    LOG(INFO) << "Printed results to " << output_file;

    std::string output_staleness = FLAGS_output_dir + "/staleness.dist";
    std::ofstream out_staleness(output_staleness);
    petuum::Table<int64_t> staleness_table =
      petuum::PSTableGroup::GetTableOrDie<int64_t>(FLAGS_staleness_table_id);
    petuum::RowAccessor row_acc;
    staleness_table.Get(0, &row_acc);
    const auto& row = row_acc.Get<petuum::DenseRow<int64_t>>();
    for (int s = 0; s < 2 * FLAGS_staleness + 1; ++s) {
      out_staleness << s - FLAGS_staleness << " " << row[s] << std::endl;
    }
    LOG(INFO) << "Printed staleness dist to " << output_staleness;
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
