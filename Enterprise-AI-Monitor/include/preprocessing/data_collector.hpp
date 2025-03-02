#pragma once
#include "monitoring_types.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>

namespace enterprise_ai {

class DataCollector {
private:
    std::queue<std::shared_ptr<MonitoringData>> data_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool running;

    // Configuration parameters
    struct Config {
        size_t queue_size_limit;
        int collection_interval_ms;
        std::string data_directory;
    } config;

public:
    DataCollector(const Config& cfg) : config(cfg), running(false) {}
    
    void start() {
        running = true;
        // Start collection threads
    }
    
    void stop() {
        running = false;
        cv.notify_all();
    }

    void processData() {
        while (running) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!data_queue.empty()) {
                auto data = data_queue.front();
                data_queue.pop();
                // Process the data
                data->preprocess();
                data->serialize();
            }
        }
    }
};

} // namespace enterprise_ai