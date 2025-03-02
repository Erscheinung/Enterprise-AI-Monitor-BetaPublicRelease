#pragma once

#include <string>
#include <vector>
#include <map>

namespace enterprise_ai {

// Data types for monitoring metrics
enum class MetricType {
    CPU_USAGE,
    MEMORY_USAGE,
    DISK_USAGE,
    NETWORK_TRAFFIC,
    SECURITY_THREAT
};

// Structure for time series data points
struct TimeSeriesPoint {
    int64_t timestamp;
    double value;
};

// Structure for a complete time series
struct TimeSeries {
    std::string metric_name;
    MetricType type;
    std::vector<TimeSeriesPoint> data_points;
};

// Structure for security threats
struct SecurityThreat {
    std::string threat_id;
    std::string threat_type;
    int severity;
    int64_t timestamp;
    std::map<std::string, std::string> metadata;
};

// Structure for resource allocation
struct ResourceAllocation {
    double cpu_percent;
    double memory_mb;
    double disk_mb;
    double network_bandwidth;
};

// Structure for a monitoring state snapshot
struct MonitoringState {
    std::map<std::string, double> resource_metrics;
    std::vector<SecurityThreat> active_threats;
    ResourceAllocation current_allocation;
};

} // namespace enterprise_ai