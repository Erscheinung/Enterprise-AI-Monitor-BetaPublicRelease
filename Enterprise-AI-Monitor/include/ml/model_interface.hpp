#pragma once
#include <string>
#include <memory>
#include <vector>
#include "../preprocessing/monitoring_types.hpp"

namespace enterprise_ai {
namespace ml {

// ID for all 3 components
enum class ModelType {
    INFRASTRUCTURE_LSTM,
    NETWORK_GNN,
    BUSINESS_RL
};

// Abstract base class defined here
class ModelInterface {
protected:
    ModelType type;
    std::string model_path;
    bool is_initialized;

public:
    ModelInterface(ModelType t, const std::string& path) 
        : type(t), model_path(path), is_initialized(false) {}
    
    virtual ~ModelInterface() = default;

    // Common core functions
    virtual bool initialize() = 0;
    virtual bool predict() = 0;
    virtual bool update() = 0;
    
    // Common utility functions
    bool isInitialized() const { return is_initialized; }
    ModelType getType() const { return type; }
};

} // namespace ml
} // namespace enterprise_ai