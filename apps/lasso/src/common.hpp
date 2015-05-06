
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
DECLARE_string(output_prefix);

DECLARE_int32(w_table_id);
DECLARE_int32(num_epochs);
DECLARE_int32(num_epochs_per_eval);
DECLARE_double(learning_rate);
