#!/usr/bin/env python

import os, sys
from os.path import dirname
from os.path import join
import subprocess

if len(sys.argv) != 2:
  print "usage: ./kill.py <hostfile>"
  sys.exit(1)

hostfile = sys.argv[1]

# Get host IPs
with open(hostfile, "r") as f:
  hostlines = f.read().splitlines()
host_ips = [line.split()[1] for line in hostlines]

ssh_cmd = (
    "ssh "
    "-o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    #"-oLogLevel=quiet "
    )

procs = []
for ip in host_ips:
  cmd = ssh_cmd + ip + " killall -q lasso_main"
  procs.append(subprocess.Popen(cmd, shell=True))
for p in procs:
  p.wait()
print "Done killing"
