#!/usr/bin/env python

import os
from os.path import dirname
from os.path import join
import time

filename = 'lasso_dim100000000_s1000000_nnz5.transx100.libsvm.trans.X'
filenameY = 'lasso_dim100000000_s1000000_nnz5.transx100.libsvm.trans.Y'
#filename = 'lasso_dim1000000000_s10000000_nnz5.transx100.libsvm.trans.X'
#filenameY = 'lasso_dim1000000000_s10000000_nnz5.transx100.libsvm.trans.Y'
#C_unsafe = 2e-3
C = 1e-4
sampling_ratio = 0.05
params = {
    "lambda": 0.1
    #, "X_file": '/users/wdai/datasets/lasso/synth/lasso_dim1000000_s500000_nnz5.transx1.libsvm.trans.X.0'
    #, "Y_file": '/users/wdai/datasets/lasso/synth/lasso_dim1000000_s500000_nnz5.transx1.libsvm.trans.Y'
    , "X_file": '/l0/data/%s' % filename
    , "Y_file": '/l0/data/%s' % filenameY
    , "global_data": "false"
    , "minibatch_ratio": sampling_ratio
    , "num_epochs": 400
    , "learning_rate": C / sampling_ratio
    , "staleness": 2
    , "num_epochs_per_eval": 12
    , "num_unused_rows": 0
    #, "num_unused_rows": 4000
    , "num_unused_cols": 1000
    , "num_partitions": 100
    , "num_partitions_per_worker": 10
    }
exp_name = "synth-100m"

hostfile_name = "nome-10" #"localserver"

app_dir = dirname(dirname(os.path.realpath(__file__)))
proj_dir = dirname(dirname(app_dir))

prog_name = "lasso_main"
prog_path = join(app_dir, "bin", prog_name)
hostfile = join(proj_dir, "machinefiles", hostfile_name)

petuum_params = {
    "hostfile": hostfile
    , "num_threads": 1
    }

env_params = (
  "GLOG_logtostderr=true "
  "GLOG_v=-1 "
  "GLOG_minloglevel=0 "
  )

ssh_cmd = (
    "ssh "
    "-o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    "-o LogLevel=quiet "
    )

# Get host IPs
with open(hostfile, "r") as f:
  #host_ips = f.read().splitlines()
  hostlines = f.read().splitlines()
host_ips = [line.split()[1] for line in hostlines]
petuum_params["num_clients"] = len(host_ips)

output_filename = exp_name
output_filename += ".N" + str(petuum_params["num_clients"])
output_filename += ".T" + str(petuum_params["num_threads"])
output_filename += ".L" + str(params["lambda"])
output_filename += ".lr" + str(params["learning_rate"])
output_filename += ".S" + str(params["staleness"])
output_filename += ".E" + str(params["num_epochs"])
output_filename += ".UR" + str(params["num_unused_rows"])
output_dir = join(app_dir, "output", output_filename)
params["output_dir"] = output_dir
cmd = "mkdir -p " + output_dir
print cmd
os.system(cmd)

for ip in host_ips:
  cmd = ssh_cmd + ip + " killall -q " + prog_name
  os.system(cmd)
print "Done killing"

for client_id, ip in enumerate(host_ips):
  cmd = ssh_cmd + ip + " "
  #cmd = ""
  cmd += env_params + prog_path
  petuum_params["client_id"] = client_id
  cmd += "".join([" --%s=%s" % (k,v) for k,v in petuum_params.items()])
  cmd += "".join([" --%s=%s" % (k,v) for k,v in params.items()])
  cmd += " &"
  print cmd
  os.system(cmd)

  if client_id == 0:
    print "Waiting for first client to set up"
    time.sleep(2)
