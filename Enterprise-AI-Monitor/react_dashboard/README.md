# Enterprise AI Monitor - C++ API Bridge

This component provides a C++ interface to the Enterprise AI Monitor system, allowing you to integrate the monitoring and analysis capabilities with C++ applications.

## Features

- Connect to the Python server API from C++ applications
- Retrieve real-time monitoring data (CPU, memory, security, costs)
- Generate advanced insights and recommendations
- Perform data analysis directly in C++
- Mock mode available when server or dependencies are unavailable

## Build Options

### Quick Start (Windows)

For Windows users, a batch file is included for easy building and running:

1. Double-click `build_and_run.bat`
2. The script will configure CMake, build the project, and run the API bridge in mock mode

### Manual Build

#### Prerequisites

- CMake 3.10 or higher
- C++ compiler with C++17 support
- Optional: libcurl development package
- Optional: LibTorch for torch integration

#### Build Steps

1. Create a build directory and navigate to it:
   ```bash
   mkdir -p build
   cd build
   ```

2. Configure with CMake:
   ```bash
   # With cURL (if available)
   cmake ..
   
   # Without cURL (mock mode)
   cmake .. -DCURL_DISABLED=ON
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

4. Run the API bridge:
   ```bash
   # On Linux/macOS
   ./api_bridge
   
   # On Windows
   .\Release\api_bridge.exe
   ```

## Usage in Your C++ Project

To integrate the API bridge into your C++ project:

1. Include the header file:
   ```cpp
   #include "api_bridge.hpp"
   ```

2. Create an instance of the bridge:
   ```cpp
   EnterpriseAIBridge bridge("http://localhost:5000/api");
   
   if (!bridge.isReady()) {
       std::cerr << "Failed to initialize API bridge" << std::endl;
       return 1;
   }
   ```

3. Use the API methods:
   ```cpp
   // Get system status
   json status = bridge.getStatus();
   
   // Start simulation if not running
   if (status["status"] != "running") {
       bridge.startSimulation();
   }
   
   // Get simulation data
   json data = bridge.getSimulationData();
   
   // Analyze the data
   bridge.analyzeData(data);
   ```

## Mock Mode

When cURL is not available or the Python server is not running, the API bridge operates in mock mode:

- Generates realistic simulation data
- Creates synthetic trends for analysis
- Provides all functionality without dependencies

To explicitly use mock mode, configure with:
```bash
cmake .. -DCURL_DISABLED=ON
```

## Dependencies

- **Required**:
  - nlohmann/json: Automatically downloaded if not found
  - C++17 standard library

- **Optional**:
  - libcurl: For HTTP requests to the Python server
  - LibTorch: For local model inference