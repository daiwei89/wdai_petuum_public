#pragma once
#include <petuum_ps_common/include/petuum_ps.hpp>
#include <string>
#include <vector>

namespace petuum {

// A singleton class.
class LossRecorder {
public:
  const static int kLossTableID;
  const static int kMaxNumFields;   // track at most this many loss fields.

  // CreateLossTable() needs to be called before CreateTableDone().
  static void CreateLossTable(int dense_float_row_type_id);

  LossRecorder();

  void RegisterField(const std::string& field_name);

  // Increment 'field_name' of ith row of loss table by val.
  void IncLoss(int ith, const std::string& field_name, float val);

  std::string PrintAllLoss();

  std::string PrintOneLoss(int ith);

private:  // private member functions
  // Return index on fields_. -1 if not found.
  int FindField(const std::string& field_name);

private:  // private member variable
  int max_eval_;   // maximum row_id that has records.
  petuum::Table<float> loss_table_;
  std::vector<std::string> fields_;

};

}   // namespace petuum
