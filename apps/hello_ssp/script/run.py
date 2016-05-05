#!/usr/bin/env python

import os
from os.path import dirname
from os.path import join
import time

params = {
    "num_iterations": 100
    , "staleness": 0
    }

hostfile_name = "cogito-2-no-yarn"

app_dir = dirname(dirname(os.path.realpath(__file__)))
proj_dir = dirname(dirname(app_dir))

prog_name = "hello_ssp"
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
  hostlines = f.read().splitlines()
host_ips = [line.split()[1] for line in hostlines]
petuum_params["num_clients"] = len(host_ips)

for ip in host_ips:
  cmd = ssh_cmd + ip + " killall -q " + prog_name
  os.system(cmd)
print "Done killing"

for client_id, ip in enumerate(host_ips):
  cmd = ssh_cmd + ip + " "
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
