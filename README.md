# Enterprise AI Monitor

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch 2.0+"/>
  <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++ 17"/>
  <img src="https://img.shields.io/badge/CMake-3.15+-green.svg" alt="CMake 3.15+"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
</div>

<p align="center">A unified AI-powered monitoring solution combining network security analysis, infrastructure prediction, and business process optimization.</p>


## 🌟 Overview

Enterprise AI Monitor is an advanced, real-time monitoring platform that leverages multiple machine learning models to provide comprehensive insights and automated responses for IT infrastructure. The system integrates three specialized AI models:

1. **Network Security Analysis**: Using Graph Neural Networks (GNN) to detect and classify security threats
2. **Infrastructure Prediction**: Employing Long Short-Term Memory networks (LSTM) to forecast system resource utilization
3. **Business Process Optimization**: Implementing Reinforcement Learning (RL) to optimize resource allocation decisions

This hybrid architecture enables:
- Predictive monitoring of system health and security threats
- Automated resource optimization based on real-time conditions
- Cost-effective infrastructure management with intelligent scaling
- Early detection of anomalies and potential security incidents

## 🚀 Features

### Real-time Monitoring Dashboard
- Interactive visualization of key metrics (CPU, memory, security threats, costs)
- Color-coded threshold zones for quick status assessment
- Responsive design with modern UI components
- Live updates with customizable refresh rate

### Advanced AI Analytics
- GNN-based security threat detection and classification
- LSTM predictive modeling for infrastructure metrics
- RL agent for autonomous decision-making and resource optimization
- Trend analysis and anomaly detection

### AI-Generated Insights
- Automatic generation of actionable insights based on current system state
- Threat level assessment and prioritized security recommendations
- Resource optimization suggestions with cost-benefit analysis
- Historical tracking of AI agent actions and their impact

### Hybrid C++/Python Architecture
- High-performance C++ core components
- Python-based ML inference server
- Seamless integration via API bridge
- Modular design for easy extension and customization

## 🏗️ System Architecture

Enterprise AI Monitor implements a three-tier architecture:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│                │      │                │      │                │
│  Web Dashboard │◄────►│  Python Server │◄────►│  C++ Core      │
│  (HTML/JS/CSS) │      │  (Flask/PyTorch)│      │  Components   │
│                │      │                │      │                │
└────────────────┘      └────────────────┘      └────────────────┘
```

### Core Components

1. **Python ML Server (python_server/)**
   - Flask-based API endpoints
   - PyTorch model loading and inference
   - Simulation engine for demo mode
   - Data processing pipeline

2. **Web Dashboard (python_server/static/)**
   - Responsive HTML/CSS interface with Bootstrap
   - Real-time data visualization with Chart.js
   - Interactive controls and insights panel
   - Action history logging

3. **C++ API Bridge (react_dashboard/src/utils/)**
   - Integration between C++ applications and Python server
   - HTTP request handling with libcurl
   - JSON data processing with nlohmann::json
   - Error handling and fallback mechanisms

4. **ML Models (pytorch models/)**
   - GNN implementation for network security (PyTorch Geometric)
   - LSTM model for infrastructure metrics prediction
   - DQN (Deep Q-Network) for reinforcement learning

## 📋 Requirements

### Python Requirements
- Python 3.8+
- PyTorch 2.0+
- Flask
- NumPy
- pandas
- scikit-learn
- torch-geometric (for GNN)

### C++ Requirements
- C++17 compatible compiler
- CMake 3.15+
- libcurl
- nlohmann::json

### Web Requirements
- Modern web browser with JavaScript enabled

## 🛠️ Installation & Setup

### Quick Start with Python Server

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/enterprise-ai-monitor.git
   cd enterprise-ai-monitor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the Python server:
   ```bash
   cd python_server
   python app.py
   ```

5. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

### Building the C++ Components

1. Ensure you have CMake and a C++17 compiler installed

2. Build the project:
   ```bash
   cmake -B build
   cmake --build build
   ```

3. Run the main application:
   ```bash
   ./build/enterprise_monitor
   ```

### Building the React Dashboard (Optional)

If you want to use the React dashboard instead of the built-in HTML dashboard:

1. Install Node.js and npm

2. Install dependencies:
   ```bash
   cd react_dashboard
   npm install
   ```

3. Build the API bridge:
   ```bash
   cd src/utils
   ./build_api_bridge.sh
   ```

4. Start the React development server:
   ```bash
   npm start
   ```

## 💻 Usage

### Starting the Monitoring System

1. Launch the Python server:
   ```bash
   cd python_server
   python app.py
   ```

2. Open the dashboard in your browser at `http://localhost:5000`

3. Click the "Start Simulation" button to begin generating and visualizing data

4. Observe the real-time metrics, AI-generated insights, and agent actions

### Understanding the Dashboard

The dashboard consists of several key components:

- **Control Panel**: Start/stop simulation and view system status
- **AI Insights Panel**: View AI-generated insights and recommendations
- **Metric Charts**: Monitor CPU, memory, security, and cost metrics in real-time
- **RL Agent Activity**: Track actions taken by the AI and their impact

### Using the C++ API Bridge

For C++ applications that need to interact with the monitoring system:

```cpp
#include "api_bridge.hpp"

int main() {
    // Initialize API bridge
    enterprise_ai::APIBridge bridge("http://localhost:5000/api");
    
    // Check server status
    auto status = bridge.getStatus();
    if (status.isRunning) {
        // Get current metrics
        auto metrics = bridge.getMetrics();
        
        // Process metrics
        std::cout << "CPU: " << metrics.cpuUtilization * 100 << "%" << std::endl;
        std::cout << "Memory: " << metrics.memoryUtilization * 100 << "%" << std::endl;
        std::cout << "Security: " << metrics.securityThreatLevel * 100 << "%" << std::endl;
    }
    
    return 0;
}
```

## 🔍 Technical Details

### ML Model Architectures

#### GNN for Network Security
- Graph Convolutional Network (GCN) implementation
- Node features representing network packet attributes
- Edge features representing connection properties
- Trained on UNSW-NB15 dataset for security analysis
- Classification of normal vs. anomalous traffic patterns

#### LSTM for Infrastructure Prediction
- Sequence-to-sequence LSTM model with attention mechanism
- Multiple input features (CPU, memory, I/O, network)
- Prediction window of 10 time steps (configurable)
- MSE loss function with Adam optimizer
- Dropout layers for regularization

#### RL for Business Process Optimization
- Deep Q-Network (DQN) with experience replay
- State space includes current system metrics and trends
- Action space includes resource allocation decisions
- Reward function balances performance and cost metrics
- ε-greedy policy with annealing for exploration

### Data Flow Pipeline

1. Raw metrics collection (real or simulated)
2. Preprocessing and normalization
3. Feature extraction and transformation
4. ML model inference and prediction
5. Decision making by RL agent
6. Action execution and impact tracking
7. Visualization and insight generation

### Extensibility

The system is designed for easy extension:

- **Add new models**: Implement the `ModelInterface` class in C++
- **Add new metrics**: Extend the data collection pipeline
- **Custom visualizations**: Modify the dashboard HTML/JS
- **Alternative backends**: Replace components while maintaining the API

## 📊 Performance Considerations

- The Python server uses multiprocessing to handle multiple ML models
- Chart.js is configured for high-performance rendering of time-series data
- API endpoints implement caching to reduce redundant computations
- RL agent uses a throttled update frequency to balance responsiveness and stability

## 🚧 Future Enhancements

Planned improvements to the system include:

1. Distributed processing for large-scale deployments
2. Integration with popular monitoring tools (Prometheus, Grafana)
3. Support for custom ML model loading
4. Enhanced anomaly detection using transfer learning
5. Multi-tenant support for enterprise deployments
6. Mobile-friendly dashboard with push notifications

## 📚 Documentation

Detailed documentation is available in the `/docs` directory:

- [API Reference](docs/api.md)
- [Model Specifications](docs/models.md)
- [Dashboard Customization](docs/dashboard.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](docs/contributing.md)

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Code of conduct
- Development workflow
- Testing requirements
- Documentation standards

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch and PyTorch Geometric teams
- UNSW-NB15 dataset creators for security data
- Chart.js project for visualization components
- Bootstrap team for UI components
- All open-source contributors whose work made this project possible
