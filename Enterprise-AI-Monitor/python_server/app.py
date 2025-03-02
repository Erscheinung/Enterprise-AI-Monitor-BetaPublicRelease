import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not available. Using mock GNN inference.")
    TORCH_GEOMETRIC_AVAILABLE = False
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
from collections import deque
import time
from threading import Thread

# Define model classes (like in our notebook)
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
   
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector

class InfrastructureLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.3):
        super(InfrastructureLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
       
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
       
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
       
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
       
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
       
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
       
    def forward(self, x):
        # Input projection
        x = self.input_layer(x)
       
        # Process LSTM layers with residual connections
        h = x
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(h)
            h = lstm_out + h  # Residual connection
            h = self.layer_norm(h)  # Layer normalization
       
        # Apply attention
        context = self.attention(h)
       
        # Output layers
        out = self.fc1(context)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
       
        return out
    
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_p=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout_p = dropout_p
   
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# Simple DQN for the RL agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
   
    def forward(self, x):
        return self.network(x)

# Business RL Agent
class BusinessRLAgent:
    def __init__(self, state_dim=4, action_dim=5, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN nets
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training params
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1  # Lower epsilon for deployment/inference
        
        # Action mapping
        self.actions = {
            0: "INCREASE_RESOURCES",
            1: "DECREASE_RESOURCES",
            2: "MAINTAIN_RESOURCES",
            3: "ENHANCE_SECURITY",
            4: "OPTIMIZE_COST"
        }
    
    def get_state(self, cpu_util, memory_util, security_threat, current_cost):
        return torch.FloatTensor([cpu_util, memory_util, security_threat, current_cost]).to(self.device)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for simulation
simulation_data = {
    "cpu_utilization": [],
    "memory_utilization": [],
    "security_threats": [],
    "costs": [],
    "actions_taken": [],
    "rewards": [],
    "timestamps": []
}

# Model paths
LSTM_MODEL_PATH = "infrastructure_lstm_model.pt"
GNN_MODEL_PATH = "gnn_model_actual_v2.pt"
RL_MODEL_PATH = "business_rl_model_final.pt"

# Model loading functions
def load_infrastructure_model(lstm_path=LSTM_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Try loading the entire model first
        model = torch.load(lstm_path, map_location=device)
        print("Loaded full LSTM model")
    except Exception as e:
        print(f"Could not load full model: {e}")
        try:
            # Try loading just the state dict
            model = InfrastructureLSTM()
            model.load_state_dict(torch.load(lstm_path, map_location=device))
            print("Loaded LSTM state dict")
        except Exception as e:
            print(f"Failed to load LSTM model: {e}")
            # Create a new model as fallback
            model = InfrastructureLSTM()
            print("Created new LSTM model as fallback")
    
    model.to(device)
    model.eval()
    return model

def load_gnn_model(gnn_path="gnn_model_actual_v2.pt", device='cpu'):
    global TORCH_GEOMETRIC_AVAILABLE
    
    try:
        # Try loading the entire model first
        if TORCH_GEOMETRIC_AVAILABLE:
            model = torch.load(gnn_path, map_location=device)
            print(f"Loaded full GNN model from {gnn_path}")
        else:
            print("Cannot load GNN model as torch_geometric is not available")
            return None
    except Exception as e:
        print(f"Could not load GNN model: {e}")
        try:
            # Try loading as state dict with a default GCN
            if TORCH_GEOMETRIC_AVAILABLE:
                model = GCN(num_features=42, hidden_channels=64, num_classes=10)
                model.load_state_dict(torch.load(gnn_path, map_location=device))
                print("Loaded GNN state dict")
            else:
                print("Cannot load GNN model as torch_geometric is not available")
                return None
        except Exception as e:
            print(f"Failed to load GNN model: {e}")
            # Create a new model as fallback
            if TORCH_GEOMETRIC_AVAILABLE:
                model = GCN(num_features=42, hidden_channels=64, num_classes=10)
                print("Created new GNN model as fallback")
            else:
                print("Cannot create GNN model as torch_geometric is not available")
                return None
    
    if model is not None:
        model.to(device)
        model.eval()
    return model

def load_rl_model(rl_path=RL_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = BusinessRLAgent()
    try:
        agent.load_model(rl_path)
    except Exception as e:
        print(f"Error loading RL model: {e}")
    return agent

def gnn_inference(input_data=None):
    global gnn_model
    
    if gnn_model is None:
        # Fall back to random values if model is not available
        return random.uniform(0, 0.2)
    
    try:
        # In a real implementation, this would be your actual GNN inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create synthetic input if none provided
        if input_data is None:
            num_nodes = 10
            num_features = 42
            
            # Generate synthetic node features
            x = torch.randn(num_nodes, num_features).to(device)
            
            # Generate a simple edge index 
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
            
            input_data = (x, edge_index)
        
        # Forward pass
        with torch.no_grad():
            x, edge_index = input_data
            logits = gnn_model(x, edge_index)
            probabilities = F.softmax(logits, dim=1)
            
            # Get the highest threat class probability
            threat_scores = []
            for i in range(probabilities.shape[1]):
                # Weight higher class indices as more severe
                severity_weight = i / (probabilities.shape[1] - 1)
                threat_scores.append(probabilities[:, i].mean().item() * severity_weight)
            
            # Normalize to [0, 1]
            total_weights = sum(i/(probabilities.shape[1] - 1) for i in range(probabilities.shape[1]))
            threat_level = sum(threat_scores) / total_weights
            return max(0.0, min(1.0, threat_level))
            
    except Exception as e:
        print(f"Error in GNN inference: {e}")
        # Return a random threat level if inference fails
        return random.uniform(0.1, 0.3)

# LSTM inference 
def lstm_inference(model, sequence_data=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate synthetic data if none provided
    if sequence_data is None:
        time_points = np.linspace(0, 2*np.pi, 10)
        base_pattern = 0.6 + 0.2 * np.sin(time_points)
        noise = np.random.normal(0, 0.05, 10)
        sequence = base_pattern + noise
        sequence = np.clip(sequence, 0, 1)
        features = np.tile(sequence.reshape(-1, 1), (1, 6))
        sequence_data = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(sequence_data)
    
    return prediction.item()

# Environment simulation for the RL agent
def simulate_environment_step(cpu_util, memory_util, security_threat, current_cost, action_idx, action_names):
    action_name = action_names[action_idx]
    
    # Base random fluctuations
    cpu_fluctuation = np.random.normal(0, 0.02)
    memory_fluctuation = np.random.normal(0, 0.02)
    security_fluctuation = np.random.normal(0, 0.01)
    cost_fluctuation = np.random.normal(0, 1.0)
    
    # Action-specific effects
    if action_name == "INCREASE_RESOURCES":
        cpu_effect = -0.15  # Reduce CPU utilization (better performance)
        memory_effect = -0.05  # Reduce memory pressure
        security_effect = -0.02  # Slight security improvement
        cost_effect = 8.0  # Higher cost
        
    elif action_name == "DECREASE_RESOURCES":
        cpu_effect = 0.15  # Increase CPU utilization (worse performance)
        memory_effect = 0.10  # Increase memory pressure
        security_effect = 0.02  # Slight security risk
        cost_effect = -6.0  # Lower cost
        
    elif action_name == "MAINTAIN_RESOURCES":
        cpu_effect = 0.0
        memory_effect = 0.0
        security_effect = 0.0
        cost_effect = 0.0
        
    elif action_name == "ENHANCE_SECURITY":
        cpu_effect = 0.05  # Slight performance impact
        memory_effect = 0.03  # Slight memory impact
        security_effect = -0.15  # Significant security improvement
        cost_effect = 5.0  # Moderate cost increase
        
    else:  # "OPTIMIZE_COST"
        cpu_effect = 0.08  # Some performance impact
        memory_effect = 0.05  # Some memory impact
        security_effect = 0.03  # Slight security impact
        cost_effect = -10.0  # Significant cost reduction
    
    # Calculate next state with both random fluctuations and action effects
    next_cpu = np.clip(cpu_util + cpu_fluctuation + cpu_effect, 0.0, 1.0)
    next_memory = np.clip(memory_util + memory_fluctuation + memory_effect, 0.0, 1.0)
    next_security = np.clip(security_threat + security_fluctuation + security_effect, 0.0, 1.0)
    next_cost = max(0, current_cost + cost_fluctuation + cost_effect)
    
    return next_cpu, next_memory, next_security, next_cost

# Function to compute reward (simplified version for simulation)
def compute_reward(state, action, next_state):
    cpu_util = state[0]
    security_threat = state[2]
    cost = state[3]
    
    next_cpu_util = next_state[0]
    next_security_threat = next_state[2]
    next_cost = next_state[3]
    
    # Base reward
    reward = 0
    
    # Resource utilization reward (prefer 60-80% CPU utilization)
    if 0.6 <= cpu_util <= 0.8:
        reward += 10
    elif cpu_util > 0.9:
        reward -= 20  # Penalize high CPU utilization
    elif cpu_util < 0.3:
        reward -= 10  # Penalize very low CPU utilization
    
    # Security threat penalties
    if security_threat > 0.7:
        reward -= 30
        
    # Cost changes
    cost_diff = next_cost - cost
    if cost_diff < 0:  # Cost decreased
        reward += min(15, abs(cost_diff))
    elif cost_diff > 0:  # Cost increased
        reward -= min(5, cost_diff)
    
    return reward

# Load models at startup
lstm_model = None
rl_agent = None
gnn_model = None

def init_models():
    global lstm_model, rl_agent
    print("Loading models...")
    lstm_model = load_infrastructure_model()
    gnn_model = load_gnn_model()
    rl_agent = load_rl_model()
    print("Models loaded")

# Simulation thread
simulation_running = False
simulation_thread = None

def run_simulation():
    global simulation_running, simulation_data
    
    # Reset simulation data
    simulation_data = {
        "cpu_utilization": [],
        "memory_utilization": [],
        "security_threats": [],
        "costs": [],
        "actions_taken": [],
        "rewards": [],
        "timestamps": []
    }
    
    # Initial state
    cpu_util = 0.7
    memory_util = 0.65
    security_threat = 0.2
    current_cost = 100.0
    
    # Simulation loop
    while simulation_running:
        # Get current timestamp
        current_time = time.time()
        
        # Get ML predictions
        predicted_cpu = lstm_inference(lstm_model)
        predicted_threat = gnn_inference()
        
        # Blend with current state
        cpu_util = 0.7 * cpu_util + 0.3 * predicted_cpu
        security_threat = 0.7 * security_threat + 0.3 * predicted_threat
        
        # Ensure values are in valid range
        cpu_util = max(0.0, min(1.0, cpu_util))
        security_threat = max(0.0, min(1.0, security_threat))
        
        # Create state for RL agent
        state = rl_agent.get_state(cpu_util, memory_util, security_threat, current_cost)
        
        # Select action
        action_idx = rl_agent.select_action(state)
        action_name = rl_agent.actions[action_idx]
        
        # Simulate environment step
        next_cpu, next_memory, next_security, next_cost = simulate_environment_step(
            cpu_util, memory_util, security_threat, current_cost, action_idx, rl_agent.actions
        )
        
        # Create next state
        next_state = rl_agent.get_state(next_cpu, next_memory, next_security, next_cost)
        
        # Compute reward
        reward = compute_reward(
            [cpu_util, memory_util, security_threat, current_cost],
            action_idx,
            [next_cpu, next_memory, next_security, next_cost]
        )
        
        # Update environment state
        cpu_util = next_cpu
        memory_util = next_memory
        security_threat = next_security
        current_cost = next_cost
        
        # Store data for monitoring
        simulation_data["cpu_utilization"].append(float(cpu_util))
        simulation_data["memory_utilization"].append(float(memory_util))
        simulation_data["security_threats"].append(float(security_threat))
        simulation_data["costs"].append(float(current_cost))
        simulation_data["actions_taken"].append(action_name)
        simulation_data["rewards"].append(float(reward))
        simulation_data["timestamps"].append(current_time)
        
        # Keep only the most recent 100 data points
        max_data_points = 100
        if len(simulation_data["timestamps"]) > max_data_points:
            for key in simulation_data:
                simulation_data[key] = simulation_data[key][-max_data_points:]
        
        # Sleep to control simulation speed
        time.sleep(1)  # Update every second

# API Routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/start-simulation', methods=['POST'])
def start_simulation():
    global simulation_running, simulation_thread
    
    if simulation_running:
        return jsonify({"status": "error", "message": "Simulation already running"})
    
    simulation_running = True
    simulation_thread = Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({"status": "success", "message": "Simulation started"})

@app.route('/api/stop-simulation', methods=['POST'])
def stop_simulation():
    global simulation_running
    
    if not simulation_running:
        return jsonify({"status": "error", "message": "No simulation running"})
    
    simulation_running = False
    
    return jsonify({"status": "success", "message": "Simulation stopped"})

@app.route('/api/simulation-data', methods=['GET'])
def get_simulation_data():
    return jsonify(simulation_data)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "running" if simulation_running else "stopped",
        "models_loaded": {
            "lstm": lstm_model is not None,
            "rl_agent": rl_agent is not None
        }
    })

# Initialize models when app starts
@app.before_first_request
def before_first_request():
    init_models()

if __name__ == '__main__':
    # Initialize models now in case we're using gunicorn or similar
    init_models()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)