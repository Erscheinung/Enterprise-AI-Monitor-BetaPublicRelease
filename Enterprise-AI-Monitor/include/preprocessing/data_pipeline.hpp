#pragma once
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include "monitoring_types.hpp"

namespace enterprise_ai {
namespace preprocessing {

class DataPipeline {
private:
    // Thread-safe queue for data processing
    std::queue<std::shared_ptr<MonitoringData>> processing_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    
    // Processing thread
    std::thread processing_thread;
    bool running;

    // Configuration
    struct Config {
        size_t batch_size;
        bool enable_augmentation;
        std::string cache_directory;
    };
    Config config;

public:
    DataPipeline(const Config& cfg) : config(cfg), running(false) {}
    
    void start() {
        running = true;
        processing_thread = std::thread([this]() {
            this->processingLoop();
        });
    }
    
    void stop() {
        running = false;
        cv.notify_all();
        if (processing_thread.joinable()) {
            processing_thread.join();
        }
    }

    void addData(std::shared_ptr<MonitoringData> data) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        processing_queue.push(data);
        cv.notify_one();
    }

private:
    void processingLoop() {
        while (running) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this]() {
                return !processing_queue.empty() || !running;
            });

            if (!running) break;

            // Process data in batches
            while (!processing_queue.empty()) {
                auto data = processing_queue.front();
                processing_queue.pop();
                
                // Preprocess based on data type
                data->preprocess();
                
                // Store processed data
                // Implementation specific to your needs
            }
        }
    }
};

} // namespace preprocessing
} // namespace enterprise_ai