
#pragma once
#include <gflags/gflags.h>

DECLARE_int32(num_clients);
DECLARE_int32(num_threads);
DECLARE_int32(client_id);
DECLARE_string(train_file);
DECLARE_int32(staleness);

DECLARE_double(lambda);
DECLARE_int32(num_samples);
DECLARE_string(X_file);
DECLARE_string(Y_file);
DECLARE_bool(global_data);
DECLARE_string(output_dir);
DECLARE_double(minibatch_ratio);
DECLARE_int32(num_partitions_per_worker);
DECLARE_int32(num_partitions);

DECLARE_int32(w_table_id);
DECLARE_int32(unused_table_id);
DECLARE_int32(staleness_table_id);
DECLARE_int32(num_epochs);
DECLARE_int32(num_epochs_per_eval);
DECLARE_double(learning_rate);
DECLARE_int32(num_unused_rows);
DECLARE_int32(num_unused_cols);
