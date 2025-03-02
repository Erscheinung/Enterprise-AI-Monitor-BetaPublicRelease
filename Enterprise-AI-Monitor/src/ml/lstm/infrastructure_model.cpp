#include "ml/lstm/infrastructure_model.hpp"
#include <stdexcept>

namespace enterprise_ai {
namespace ml {

InfrastructureLSTM::InfrastructureLSTM(const std::string& model_path)
    : ModelInterface(ModelType::INFRASTRUCTURE_LSTM, model_path),
      device(torch::kCPU)
{
    // Choose GPU if available
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA for Infrastructure LSTM" << std::endl;
    } else {
        std::cout << "Using CPU for Infrastructure LSTM" << std::endl;
    }
}

bool InfrastructureLSTM::initialize() {
    try {
        // Load the JIT-compiled PyTorch model
        model = torch::jit::load(model_path);
        model.to(device);
        // Set the model to evaluation mode
        model.eval();
        is_initialized = true;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl; // Changed e.msg() to e.what()
        return false;
    }
}

bool InfrastructureLSTM::predict() {
    // We can only predict if the model is initialized and we have enough data
    if (!is_initialized || metrics_buffer.size() < sequence_length) {
        std::cerr << "Model not initialized or insufficient data in buffer.\n";
        return false;
    }

    try {
        // Prepare input data from the last 'sequence_length' metrics
        std::vector<float> input_data;
        input_data.reserve(sequence_length * 3); // change this to 6 later as we used 6

        size_t start_idx = metrics_buffer.size() - sequence_length;
        for (size_t i = start_idx; i < metrics_buffer.size(); ++i) {
            float n_cpu = static_cast<float>(normalize(metrics_buffer[i].cpu_usage, cpu_mean, cpu_std));
            float n_mem = static_cast<float>(normalize(metrics_buffer[i].memory_usage, mem_mean, mem_std));
            float n_io  = static_cast<float>(normalize(metrics_buffer[i].io_rate, io_mean, io_std));
            input_data.push_back(n_cpu);
            input_data.push_back(n_mem);
            input_data.push_back(n_io);
        }

        // Create a tensor of shape [batch_size=1, seq_len=10, features=3] - change features to 6 !!!
        auto input_tensor = torch::from_blob(
            input_data.data(),
            {1, static_cast<long>(sequence_length), 3},
            torch::TensorOptions().dtype(torch::kFloat)
        ).clone().to(device);

        // Prepare inputs for the model forward
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Maybe wrap in a no-grad guard
        auto output = model.forward(inputs).toTensor();

        // Do something with 'output' - CPU % preds
        if (output.sizes().size() > 0) {
            auto predicted_value = output.item<float>();
            std::cout << "Predicted value: " << predicted_value << std::endl;
        }

        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        return false;
    }
}

bool InfrastructureLSTM::update() {
    return true;
}

void InfrastructureLSTM::addMetrics(const InfrastructureMetrics& metrics) {
    metrics_buffer.push_back(metrics);
}

double InfrastructureLSTM::normalize(double value, double mean, double std) {
    // divide by 0
    return (std == 0.0) ? (value - mean) : (value - mean) / std;
}

} // namespace ml
} // namespace enterprise_ai
