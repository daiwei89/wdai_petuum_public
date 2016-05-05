
#pragma once

#include <map>
#include <string>

#include <petuum_ps_common/include/host_info.hpp>

namespace petuum {
// Read in a file containing list of servers. 'server_file' need to have the
// following line structure:
//
// <id> <ip> <port> (tab in as deliminator) 1 128.0.1.1 80
//
// Note that the first line of the file will be considered as name node.
void GetHostInfos(std::string server_file,
  std::map<int32_t, HostInfo> *host_map);

// Read in a file containing list of servers IPs (in separate lines) and
// use default port.
// Note that the first line of the file will be considered as name node.
std::map<int32_t, HostInfo> GetHostInfosSimple(const std::string& server_file);

void GetServerIDsFromHostMap(std::vector<int32_t> *server_ids,
  const std::map<int32_t, HostInfo> & host_map);

}   // namespace petuum
