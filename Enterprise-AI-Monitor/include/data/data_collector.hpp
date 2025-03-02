// include/data/collector.hpp
#pragma once
#include <memory>
#include <queue>
#include <mutex>
#include "monitoring_types.hpp"
#include "ml/lstm/infrastructure_model.hpp"
#include "ml/gnn/network_model.hpp"

namespace enterprise_ai {
namespace data {

class UnifiedDataCollector {
private:
    // Shared pointers to our ML models
    std::shared_ptr<ml::InfrastructureLSTM> infra_model;
    std::shared_ptr<ml::NetworkGNN> network_model;
    
    // Thread-safe queues for different data types
    std::queue<ml::InfrastructureMetrics> infra_queue;
    std::queue<ml::NetworkGraph> network_queue;
    
    // Synchronization primitives
    std::mutex infra_mutex;
    std::mutex network_mutex;
    std::condition_variable cv;
    
    bool running;
    
    // Configuration parameters
    struct Config {
        size_t queue_size_limit;
        int collection_interval_ms;
        std::string data_directory;
    } config;

public:
    UnifiedDataCollector(const Config& cfg,
                        std::shared_ptr<ml::InfrastructureLSTM> infra,
                        std::shared_ptr<ml::NetworkGNN> network);
    
    void start();
    void stop();
    
    void addInfrastructureData(const ml::InfrastructureMetrics& metrics);
    void addNetworkData(const ml::NetworkGraph& graph);
    
private:
    void processInfrastructureData();
    void processNetworkData();
};

} // namespace data
} // namespace enterprise_ai