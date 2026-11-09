/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * KV offload prototype is licensed under Mulan PSL v2.
 * emsd 入口。
 *
 * 用法：
 *   master: emsd --role master --ip <本机IP> --port 27000 \
 *               --members <ip1:port1:hcommPort1,ip2:port2:hcommPort2,...> [--pool-size-gb 8]
 *   worker: emsd --role worker --ip <本机IP> --port 27001 \
 *               --master <ip:port>   （worker 启动即向 master 注册，静态成员表由此生成）
 *
 * 说明：静态成员表以 master 的 --members 为准；worker --master 仅用于自检连通性（可选）。
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ctrl_server.h"

namespace {
void Usage(const char *bin)
{
    printf("usage: %s --role master|worker --ip <localIp> --port <ctrlPort>\n"
           "  [--device N] [--rootinfo <path>] [--acl-lib <path>] [--hcomm-lib <path>] [--pool-size-gb N]\n"
           "  master: --members ip1:port1:hcommPort1[,ip2:port2:hcommPort2...]\n"
           "  EID: 自动按 rootinfo（默认 /etc/hccl_rootinfo.json）匹配 device_id==phyId 的 EID\n"
           "env: KV_EMSD_HCOMM_LIB 覆盖 libhcomm 路径\n",
           bin);
}

bool ParseHostPort(const std::string &s, std::string &ip, uint16_t &port)
{
    auto pos = s.rfind(':');
    if (pos == std::string::npos) {
        return false;
    }
    ip = s.substr(0, pos);
    port = static_cast<uint16_t>(atoi(s.substr(pos + 1).c_str()));
    return !ip.empty() && port != 0;
}
}  // namespace

int main(int argc, char **argv)
{
    EmsdConfig cfg;
    std::string role = "worker";
    std::string members;
    int poolGb = 8;
    int deviceId = 0;

    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--role") {
            role = next();
        } else if (k == "--ip") {
            cfg.localIp = next();
        } else if (k == "--port") {
            cfg.listenPort = static_cast<uint16_t>(atoi(next().c_str()));
        } else if (k == "--members") {
            members = next();
        } else if (k == "--pool-size-gb") {
            poolGb = atoi(next().c_str());
        } else if (k == "--device") {
            deviceId = atoi(next().c_str());
        } else if (k == "--rootinfo") {
            cfg.rootinfoPath = next();
        } else if (k == "--acl-lib") {
            cfg.aclLib = next();
        } else if (k == "--hcomm-lib") {
            cfg.hcommLib = next();
        } else {
            Usage(argv[0]);
            return 1;
        }
    }
    if (role != "master" && role != "worker") {
        Usage(argv[0]);
        return 1;
    }
    if (const char *env = getenv("KV_EMSD_HCOMM_LIB"); env != nullptr && cfg.hcommLib == "libhcomm.so") {
        cfg.hcommLib = env;
    }
    cfg.isMaster = (role == "master");
    cfg.poolSizeBytes = static_cast<uint64_t>(poolGb) << 30;
    cfg.deviceId = static_cast<int32_t>(deviceId);
    if (cfg.localIp.empty()) {
        printf("[emsd][ERROR] --ip is required\n");
        Usage(argv[0]);
        return 1;
    }

    if (cfg.isMaster) {
        if (members.empty()) {
            printf("[emsd][ERROR] master requires --members\n");
            return 1;
        }
        uint16_t baseHcommPort = 60001;
        size_t start = 0;
        uint32_t nodeId = 0;
        while (start <= members.size()) {
            auto end = members.find(',', start);
            std::string item = members.substr(start, (end == std::string::npos ? members.size() : end) - start);
            if (!item.empty()) {
                // item = ip:ctrlPort[:hcommPort]
                auto p1 = item.find(':');
                auto p2 = item.find(':', p1 + 1);
                EmsdMember m{};
                m.nodeId = nodeId;
                m.ip = item.substr(0, p1);
                m.ctrlPort = static_cast<uint16_t>(atoi(item.substr(p1 + 1, p2 - p1 - 1).c_str()));
                m.hcommPort = (p2 == std::string::npos) ? static_cast<uint16_t>(baseHcommPort + nodeId)
                                                        : static_cast<uint16_t>(atoi(item.substr(p2 + 1).c_str()));
                cfg.members.push_back(m);
                nodeId++;
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        if (cfg.members.empty() || cfg.members.size() > KV_MAX_SEG_NUM) {
            printf("[emsd][ERROR] invalid --members (1..%u nodes)\n", KV_MAX_SEG_NUM);
            return 1;
        }
    } else {
        EmsdMember self{};
        self.nodeId = 0;
        self.ip = cfg.localIp;
        self.ctrlPort = cfg.listenPort;
        self.hcommPort = 60001;
        cfg.members.push_back(self);  // worker 自身段表仅一条
    }

    printf("[emsd] starting, role=%s ip=%s port=%u device=%d rootinfo=%s poolSizeBytes=%llu aclLib=%s "
           "hcommLib=%s\n",
           cfg.isMaster ? "master" : "worker", cfg.localIp.c_str(), cfg.listenPort, cfg.deviceId,
           cfg.rootinfoPath.c_str(), (unsigned long long)cfg.poolSizeBytes, cfg.aclLib.c_str(),
           cfg.hcommLib.c_str());
    EmsdCtrlServer server(cfg);
    return server.Run();
}
