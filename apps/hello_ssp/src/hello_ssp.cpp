#include <petuum_ps_common/include/petuum_ps.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>
#include <cstdint>
#include <random>
#include <chrono>

DEFINE_string(hostfile, "", "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, "# of nodes");
DEFINE_int32(client_id, 0, "Client ID");
DEFINE_int32(num_threads, 1, "# of worker threads.");
DEFINE_int32(num_iterations, 10, "# of iterations.");
DEFINE_int32(staleness, 0, "Staleness.");

DEFINE_int32(max_delay_millisec, 100, "");

// HelloSSP demonstrates the concept of stale synchronous parallel (SSP). It
// uses 1 table with 1 row, with num_workers columns (FLAGS_num_clients *
// FLAGS_num_threads), each entry is the clock of one worker as viewed by each
// worker. SSP ensures that the clock view will be at most 'staleness' clocks
// apart, which we verify here.

namespace {

const int kDenseRowType = 0;
const int kTableID = 0;

void HelloSSPWorker(int worker_rank) {
  petuum::PSTableGroup::RegisterThread();
  std::random_device rd;
  std::mt19937 rng;
  std::uniform_int_distribution<int> dist(0,FLAGS_max_delay_millisec);
  petuum::Table<double> clock_table =
    petuum::PSTableGroup::GetTableOrDie<double>(kTableID);
  int num_workers = FLAGS_num_clients * FLAGS_num_threads;
  std::vector<double> row_cache;
  // Register callbacks.
  clock_table.GetAsyncForced(0);
  clock_table.WaitPendingAsyncGet();
  for (int clock = 0; clock < FLAGS_num_iterations; ++clock) {
    std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
    petuum::RowAccessor row_acc;
    clock_table.Get(0, &row_acc);
    const auto& row = row_acc.Get<petuum::DenseRow<double>>();
    // Cache the row to a local vector to avoid acquiring lock when accessing
    // each entry.
    row.CopyToVector(&row_cache);
    for (int w = 0; w < num_workers; ++w) {
      CHECK_GE(row_cache[w], clock - FLAGS_staleness) << "I'm worker "
        << worker_rank << " seeing w: " << w;
    }
    // Update clock entry associated with this worker.
    clock_table.Inc(0, worker_rank, 1);
    petuum::PSTableGroup::Clock();
  }

  // After global barrier, all clock view should be up-to-date.
  petuum::PSTableGroup::GlobalBarrier();
  petuum::RowAccessor row_acc;
  clock_table.Get(0, &row_acc);
  const auto& row = row_acc.Get<petuum::DenseRow<double>>();
  row.CopyToVector(&row_cache);
  for (int w = 0; w < num_workers; ++w) {
    CHECK_EQ(row_cache[w], FLAGS_num_iterations);
  }
  LOG(INFO) << "Worker " << worker_rank << " verified all clock reads.";
  petuum::PSTableGroup::DeregisterThread();
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  petuum::TableGroupConfig table_group_config;
  table_group_config.num_comm_channels_per_client = 1;
  table_group_config.num_total_clients = FLAGS_num_clients;
  table_group_config.num_tables = 1;    // just the clock table
  // + 1 for main() thread.
  table_group_config.num_local_app_threads = FLAGS_num_threads + 1;
  table_group_config.client_id = FLAGS_client_id;
  table_group_config.consistency_model = petuum::SSPPush;

  petuum::GetHostInfos(FLAGS_hostfile, &table_group_config.host_map);

  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<double>>(kDenseRowType);

  petuum::PSTableGroup::Init(table_group_config, false);

  // Creating test table.
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_type = kDenseRowType;
  table_config.table_info.table_staleness = FLAGS_staleness;
  table_config.table_info.row_capacity = FLAGS_num_clients * FLAGS_num_threads;
  table_config.process_cache_capacity = 1;    // only 1 row.
  table_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  // no need to squeeze out 0's
  table_config.table_info.oplog_dense_serialized = false;
  table_config.table_info.dense_row_oplog_capacity =
    table_config.table_info.row_capacity;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  CHECK(petuum::PSTableGroup::CreateTable(kTableID, table_config));

  petuum::PSTableGroup::CreateTableDone();

  LOG(INFO) << "Starting HelloSSP with " << FLAGS_num_threads << " threads "
    << "on client " << FLAGS_client_id;

  std::vector<std::thread> threads(FLAGS_num_threads);
  int worker_rank = FLAGS_client_id * FLAGS_num_threads;
  for (auto& thr : threads) {
    thr = std::thread(HelloSSPWorker, worker_rank++);
  }
  for (auto& thr : threads) {
    thr.join();
  }

  petuum::PSTableGroup::ShutDown();
  LOG(INFO) << "HelloSSP Finished!";
  return 0;
}
