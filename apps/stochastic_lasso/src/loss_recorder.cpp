#include "loss_recorder.hpp"
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <algorithm>
#include <glog/logging.h>
#include <cmath>
#include <sstream>

namespace petuum {

const int LossRecorder::kLossTableID = 999;
const int LossRecorder::kMaxNumFields = 100;

namespace {

const int kProcessCacheSize = 1000;

}  // anonymous namespace

void LossRecorder::CreateLossTable(int dense_float_row_type_id) {
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_type = dense_float_row_type_id;
  table_config.table_info.table_staleness = 0;
  table_config.table_info.row_capacity = kMaxNumFields;
  table_config.process_cache_capacity = kProcessCacheSize;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  table_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  // no need to squeeze out 0's.
  table_config.table_info.oplog_dense_serialized = false;
  table_config.table_info.dense_row_oplog_capacity = kMaxNumFields;
  petuum::PSTableGroup::CreateTable(kLossTableID, table_config);
}

LossRecorder::LossRecorder() : max_eval_(0) {
  loss_table_ =
    petuum::PSTableGroup::GetTableOrDie<float>(kLossTableID);
}

void LossRecorder::RegisterField(const std::string& field_name) {
  fields_.push_back(field_name);
}

int LossRecorder::FindField(const std::string& field_name) {
  for (int i = 0; i < fields_.size(); ++i) {
    if (fields_[i] == field_name) {
      return i;
    }
  }
  return -1;
}

void LossRecorder::IncLoss(int ith, const std::string& field_name, float val) {
  CHECK_GE(ith, 0);
  CHECK_LT(ith, kProcessCacheSize);
  int field_id = FindField(field_name);
  CHECK_NE(-1, field_id) << "Field " << field_name << " not found.";
  max_eval_ = std::max(ith, max_eval_);
  loss_table_.Inc(ith, field_id, val);
}

std::string LossRecorder::PrintAllLoss() {
  std::stringstream ss;
  for (int j = 0; j < fields_.size(); ++j) {
    ss << fields_[j] << " ";
  }
  ss << "\n";

  // Print each row.
  for (int i = 0; i <= max_eval_; ++i) {
    petuum::RowAccessor row_acc;
    loss_table_.Get(i, &row_acc);
    const auto& loss_row = row_acc.Get<petuum::DenseRow<float>>();
    for (int j = 0; j < fields_.size(); ++j) {
      ss << loss_row[j] << " ";
      //CHECK(!std::isnan(loss_row[j]));
    }
    ss << "\n";
  }
  return ss.str();
}

std::string LossRecorder::PrintOneLoss(int ith) {
  CHECK_GE(ith, 0);
  CHECK_LT(ith, kProcessCacheSize);
  petuum::RowAccessor row_acc;
  loss_table_.Get(ith, &row_acc);
  const auto& loss_row = row_acc.Get<petuum::DenseRow<float>>();
  std::stringstream ss;
  for (int j = 0; j < fields_.size(); ++j) {
    ss << fields_[j] << " " << loss_row[j] << " ";
  }
  return ss.str();
}

}  // namespace petuum
