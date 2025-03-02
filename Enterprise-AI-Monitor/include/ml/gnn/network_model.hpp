// include/ml/gnn/network_model.hpp
#pragma once
#include "ml/model_interface.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>

namespace enterprise_ai {
namespace ml {

struct NetworkNode {
    std::string ip_address;
    std::vector<float> features;  // Node features -> packet counts, byte counts
    
    // Custom hash function for network nodes
    bool operator==(const NetworkNode& other) const {
        return ip_address == other.ip_address;
    }
};

struct NetworkEdge {
    NetworkNode source;
    NetworkNode target;
    std::vector<float> features;  // Edge features like protocol, duration
};

struct NetworkGraph {
    std::vector<NetworkNode> nodes;
    std::vector<NetworkEdge> edges;
    timestamp_t timestamp;
};

class NetworkGNN : public ModelInterface {
private:
    torch::jit::script::Module model;
    torch::Device device;
    
    // GNN parameters
    size_t node_feature_dim;
    size_t edge_feature_dim;
    size_t hidden_dim;
    
    // To perform temporal analysis
    std::vector<NetworkGraph> graph_buffer;

public:
    NetworkGNN(const std::string& model_path);
    ~NetworkGNN() override = default;

    bool initialize() override;
    bool predict() override;
    bool update() override;

private:
    torch::Tensor createGraphBatch(const std::vector<NetworkGraph>& graphs);
    void normalizeFeatures(NetworkGraph& graph);
};

}
}