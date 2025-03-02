// include/ml/lstm/infrastructure_model.hpp
#pragma once
#include "../model_interface.hpp"
#include <torch/torch.h>
#include <torch/script.h> 
#include <string>
#include <vector>
#include <iostream>
#include <memory>

// Use timestamp_t from monitoring_types.hpp or define globally
using timestamp_t = uint64_t;

namespace enterprise_ai {
namespace ml {

// Remove ModelType and ModelInterface redefinitions

//------------------------------------------------------------------
// Definition of a metrics struct
//------------------------------------------------------------------
struct InfrastructureMetrics {
    double cpu_usage;
    double memory_usage;
    double io_rate;
    timestamp_t timestamp;
};

//------------------------------------------------------------------
// Our InfrastructureLSTM class
//------------------------------------------------------------------
class InfrastructureLSTM : public ModelInterface {
private:
    torch::jit::script::Module model;
    torch::Device device;
    // normalization - means and stddevs
    double cpu_mean = 0.0, cpu_std = 1.0;
    double mem_mean = 0.0, mem_std = 1.0;
    double io_mean = 0.0, io_std = 1.0;
    // LSTM seq length
    const size_t sequence_length = 10;
    // Buffer for latest metrics
    std::vector<InfrastructureMetrics> metrics_buffer;

public:
    // Constructor 
    InfrastructureLSTM(const std::string& model_path);
    ~InfrastructureLSTM() override = default;
    
    // Inherited interface methods
    bool initialize() override;
    bool predict() override;
    bool update() override;
    
    // A method to push new data into the metrics buffer (example)
    void addMetrics(const InfrastructureMetrics& metrics);

private:
    // Normalization helper
    double normalize(double value, double mean, double std);
};

} // namespace ml
} // namespace enterprise_ai