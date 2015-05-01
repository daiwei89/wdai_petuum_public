#include "lasso_engine.hpp"
#include "loss_recorder.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>
#include <vector>
#include <cstdint>

// Petuum Parameters
DEFINE_string(hostfile, "", "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, "Total number of clients");
DEFINE_int32(num_threads, 1, "Number of app threads in this client");
DEFINE_int32(client_id, 0, "Client ID");
DEFINE_string(consistency_model, "SSPPush", "SSP or SSPPush");
DEFINE_int32(num_comm_channels_per_client, 1,
    "number of comm channels per client");
DEFINE_int32(staleness, 0, "SSP staleness");

// Lasso Parameters
DEFINE_double(lambda, 0.1, "regularization param.");
DEFINE_string(X_file, "", "The program expects 2 files: X_file, "
    "X_file.meta. If global_data = false, then it looks for X_file.X, "
    "X_file.X.meta, where X is the client_id.");
DEFINE_string(Y_file, "", "Each line is a label y.");
DEFINE_bool(global_data, false, "If true, all workers read from the same "
    "train_file. If false, append X. See train_file.");
DEFINE_int32(num_epochs, 10, "# of data passes.");
DEFINE_double(learning_rate, 0.1, "Initial step size");

// Misc
DEFINE_int32(w_table_id, 0, "Weight table's ID in PS.");
DEFINE_int32(loss_table_id, 1, "Loss table's ID in PS.");
DEFINE_int32(row_oplog_type, petuum::RowOpLogType::kDenseRowOpLog,
    "row oplog type");
DEFINE_bool(oplog_dense_serialized, false, "True to not squeeze out the 0's "
    "in dense oplog.");

const int32_t kDenseRowFloatTypeID = 0;

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  petuum::TableGroupConfig table_group_config;
  table_group_config.num_comm_channels_per_client
    = FLAGS_num_comm_channels_per_client;
  table_group_config.num_total_clients = FLAGS_num_clients;
  // w table and a loss table.
  table_group_config.num_tables = 2;
  // + 1 for main() thread.
  table_group_config.num_local_app_threads = FLAGS_num_threads + 1;
  table_group_config.client_id = FLAGS_client_id;

  petuum::GetHostInfos(FLAGS_hostfile, &table_group_config.host_map);
  if (std::string("SSP").compare(FLAGS_consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSP;
  } else if (std::string("SSPPush").compare(FLAGS_consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSPPush;
  } else if (std::string("LocalOOC").compare(FLAGS_consistency_model) == 0) {
    table_group_config.consistency_model = petuum::LocalOOC;
  } else {
    LOG(FATAL) << "Unkown consistency model: " << FLAGS_consistency_model;
  }

  lasso::LassoEngine lasso_engine;
  int num_samples = lasso_engine.ReadData();

  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float>>
    (kDenseRowFloatTypeID);

  // Use false to disallow main thread to access table API.
  petuum::PSTableGroup::Init(table_group_config, false);

  petuum::ClientTableConfig table_config;
  table_config.table_info.row_type = kDenseRowFloatTypeID;
  table_config.table_info.table_staleness = FLAGS_staleness;
  // Each worker needs a row.
  table_config.table_info.row_capacity = num_samples;
  table_config.table_info.row_oplog_type = FLAGS_row_oplog_type;
  table_config.table_info.oplog_dense_serialized =
    FLAGS_oplog_dense_serialized;
  table_config.table_info.dense_row_oplog_capacity = num_samples;
  table_config.process_cache_capacity =
    FLAGS_num_clients * FLAGS_num_threads + 1;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(FLAGS_w_table_id, table_config);

  petuum::LossRecorder::CreateLossTable(kDenseRowFloatTypeID);

  petuum::PSTableGroup::CreateTableDone();

  LOG(INFO) << "Starting Lasso with " << FLAGS_num_threads << " threads "
    << "on client " << FLAGS_client_id;

  std::vector<std::thread> threads(FLAGS_num_threads);
  for (auto& thr : threads) {
    thr = std::thread(&lasso::LassoEngine::Start, std::ref(lasso_engine));
  }
  for (auto& thr : threads) {
    thr.join();
  }

  petuum::PSTableGroup::ShutDown();
  LOG(INFO) << "Lasso finished and shut down!";
  return 0;
}
