#ifndef API_BRIDGE_HPP
#define API_BRIDGE_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// API Bridge class declaration
class EnterpriseAIBridge {
private:
    std::string baseUrl;
    void* curl;
    bool isConnected;

    // Helper methods to perform HTTP requests
    bool performGet(const std::string& url, std::string& response);
    bool performPost(const std::string& url, const std::string& data, std::string& response);
    double calculateAverage(const std::vector<double>& values);
    
    // Data generation for mock mode
    json generateMockData();
    
    // Insight generation
    void generateInsights(
        const std::vector<double>& cpuData,
        const std::vector<double>& memoryData,
        const std::vector<double>& securityData,
        const std::vector<double>& costData
    );
    
    // LibTorch integration method
    void analyzeWithLibTorch(
        const std::vector<double>& cpuData,
        const std::vector<double>& memoryData,
        const std::vector<double>& securityData,
        const std::vector<double>& costData
    );

public:
    // Constructor & destructor
    EnterpriseAIBridge(const std::string& url = "http://localhost:5000/api");
    ~EnterpriseAIBridge();

    // API methods
    bool isReady() const;
    json getStatus();
    json getSimulationData();
    bool startSimulation();
    bool stopSimulation();
    void analyzeData(const json& data);
};

#endif // API_BRIDGE_HPP