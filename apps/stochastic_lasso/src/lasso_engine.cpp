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
#include <gflags/gflags.h>


namespace lasso {

LassoEngine::LassoEngine() : thread_counter_(0) {
  // Apepnd client_id if the train_data isn't global.
  if (FLAGS_global_data) {
    std::string meta_file = FLAGS_X_file + ".meta";
    petuum::ml::MetafileReader mreader(meta_file);
    // num features in the X_file (# of rows)
    num_features_ += mreader.get_int32("num_features");
    // num_samples_ is the dim of each feature column.
    num_samples_ = mreader.get_int32("num_samples");
    Y_ = petuum::ml::DenseFeature<float>(num_samples_);
    CHECK_EQ("libsvm", mreader.get_string("format")) << "Only support libsvm";
    sample_one_based_ = mreader.get_bool("sample_one_based");
    snappy_compressed_ = mreader.get_bool("snappy_compressed");
  } else {
    par_begin_ = FLAGS_num_partitions_per_worker * FLAGS_client_id;
    par_end_ = FLAGS_client_id == FLAGS_num_clients - 1 ? 
      FLAGS_num_partitions : par_begin_ + FLAGS_num_partitions_per_worker;
    LOG(INFO) << "worker " << FLAGS_client_id << " par_begin: " << par_begin_
      << " par_end: " << par_end_;
    for (int i = par_begin_; i < par_end_; ++i) {
      std::string meta_file = FLAGS_X_file
        +  "." + std::to_string(i) + ".meta";
      petuum::ml::MetafileReader mreader(meta_file);
      // num features in the X_file (# of rows)
      num_features_ += mreader.get_int32("num_features_this_partition");
      // num_samples_ is the dim of each feature column.
      num_samples_ = mreader.get_int32("num_samples");
      CHECK_EQ("libsvm", mreader.get_string("format")) << "Only support libsvm";
      sample_one_based_ = mreader.get_bool("sample_one_based");
      snappy_compressed_ = mreader.get_bool("snappy_compressed");
    }
    Y_ = petuum::ml::DenseFeature<float>(num_samples_);
  }
}

LassoEngine::~LassoEngine() {
  for (auto& row_ptr : X_cols_) {
    delete row_ptr;
  }
}

int LassoEngine::ReadData() {
  if (FLAGS_global_data) {
    std::string X_file = FLAGS_X_file;
    LOG(INFO) << "Reading X_file: " << X_file;
    // Note feature and sample are transposed (each row is a feature)
    std::vector<int> feature_idx;
    petuum::ml::ReadDataLabelLibSVM(X_file, num_samples_, num_features_,
        reinterpret_cast<std::vector<
        petuum::ml::AbstractFeature<float>*>*>(&X_cols_),
        &feature_idx, sample_one_based_, false,  // label is real value
        snappy_compressed_);
  } else {
    for (int i = par_begin_; i < par_end_; ++i) {
      std::string X_file = FLAGS_X_file + "." + std::to_string(i);
      LOG(INFO) << "Reading X_file: " << X_file;

      std::string meta_file = FLAGS_X_file
        +  "." + std::to_string(i) + ".meta";
      petuum::ml::MetafileReader mreader(meta_file);
      // num features in the X_file (# of rows)
      int num_features_this_partition =
        mreader.get_int32("num_features_this_partition");
      // num_samples_ is the dim of each feature column.
      int num_samples_this_partition = mreader.get_int32("num_samples");
      // Note feature and sample are transposed (each row is a feature)
      std::vector<int> feature_idx;
      std::vector<petuum::ml::SparseFeature<float>*> X_cols_this_partition;
      petuum::ml::ReadDataLabelLibSVM(X_file, num_samples_this_partition,
          num_features_this_partition,
          reinterpret_cast<std::vector<
          petuum::ml::AbstractFeature<float>*>*>(&X_cols_this_partition),
          &feature_idx, sample_one_based_, false,  // label is real value
          snappy_compressed_);
      for (int j = 0; j < X_cols_this_partition.size(); ++j) {
        // This transfer ownership of SparseFeature<float>* from
        // X_cols_this_partition to X_cols_.
        X_cols_.push_back(X_cols_this_partition[j]);
      }
    }
  }
  CHECK_EQ(num_features_, X_cols_.size());

  // Read y (always global).
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
  LOG(INFO) << "Worker " << FLAGS_client_id << " read data: num_features: "
    << num_features_ << " num_samples: " << num_samples_;
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

  int global_worker_rank = FLAGS_client_id * FLAGS_num_threads + thread_id;
  int worker_rank = FLAGS_global_data ?
    global_worker_rank : thread_id;
  int num_workers = FLAGS_global_data ?
    FLAGS_num_clients * FLAGS_num_threads : FLAGS_num_threads;
  int num_feature_per_worker = num_features_ / num_workers;
  int feature_start = worker_rank * num_feature_per_worker;
  int last_worker_rank = FLAGS_global_data ?
    FLAGS_num_clients * FLAGS_num_threads - 1 : FLAGS_num_threads - 1;
  int feature_end = (worker_rank == last_worker_rank)
    ? num_features_ : feature_start + num_feature_per_worker;

  ProxGradConfig pg_config;
  pg_config.worker_rank = global_worker_rank;
  pg_config.feature_start = feature_start;
  pg_config.feature_end = feature_end;
  pg_config.num_samples = num_samples_;
  ProxGrad pg(pg_config);

  int eval_counter = 0;
  for (int epoch = 1; epoch <= FLAGS_num_epochs; ++epoch) {
    // Constant learning rate.
    pg.ProxStep(X_cols_, Y_, FLAGS_learning_rate, epoch - 1);

    // Evaluate objective value.
    if (epoch == 1 || epoch % FLAGS_num_epochs_per_eval == 0) {
      //LOG(INFO) << "evaluating...";
      loss_recorder.IncLoss(eval_counter, "FullLoss", pg.EvalL1Penalty());
      loss_recorder.IncLoss(eval_counter, "BetaNNZ", pg.EvalBetaNNZ());
      if (global_worker_rank == 0) {
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

  if (global_worker_rank == 0) {
    LOG(INFO) << "\n" << PrintExpDetail(pg) << loss_recorder.PrintAllLoss();
    std::string output_file = FLAGS_output_dir + "/loss";
    std::ofstream out(output_file);
    out << PrintExpDetail(pg) << loss_recorder.PrintAllLoss();
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

std::string LassoEngine::PrintExpDetail(const ProxGrad& pg) const {
  std::stringstream ss;
  const auto& flags = google::GetArgvs();
  for (int i = 0; i < flags.size(); ++i) {
    if (i == 0) {
      continue;   // skip the program name.
    }
    // s looks like --num_threads=16
    const auto& s = flags[i];
    std::size_t pos = s.find('=');
    ss << s.substr(2, pos - 2);   // 2 to skip "--"
    ss << ": " << s.substr(pos + 1) << std::endl;
  }
  ss << "sampled dim: " << pg.GetSampleSize() << std::endl;
  int num_workers = FLAGS_global_data ?
    FLAGS_num_clients * FLAGS_num_threads : FLAGS_num_threads;
  int num_features_per_worker = num_features_ / num_workers;
  ss << "num dim this worker: " << num_features_per_worker << std::endl;
  ss << "sampling ratio: " << static_cast<float>(pg.GetSampleSize())
    / num_features_per_worker << std::endl;
  return ss.str();
}

}  // namespace lasso
