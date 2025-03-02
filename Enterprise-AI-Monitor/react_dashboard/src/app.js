import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer 
} from 'recharts';
import axios from 'axios';
import { 
  Container, Grid, Paper, Typography, Button, Box, 
  Card, CardContent, CardHeader, CircularProgress, 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow 
} from '@mui/material';
import './App.css';

// API base URL
const API_URL = 'http://localhost:5000/api';

function App() {
  // State variables
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationData, setSimulationData] = useState({
    cpu_utilization: [],
    memory_utilization: [],
    security_threats: [],
    costs: [],
    actions_taken: [],
    rewards: [],
    timestamps: []
  });
  const [statusInfo, setStatusInfo] = useState({
    models_loaded: {
      lstm: false,
      rl_agent: false
    }
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Format data for charts
  const formatChartData = () => {
    const { 
      cpu_utilization, memory_utilization, security_threats, 
      costs, timestamps, actions_taken, rewards 
    } = simulationData;
    
    // If no data yet, return empty array
    if (!timestamps || timestamps.length === 0) return [];
    
    // Create formatted data for charts
    return timestamps.map((timestamp, index) => {
      const formattedTime = new Date(timestamp * 1000).toLocaleTimeString();
      return {
        time: formattedTime,
        cpu: cpu_utilization[index] * 100, // Convert to percentage
        memory: memory_utilization[index] * 100, // Convert to percentage
        security: security_threats[index] * 100, // Convert to percentage
        cost: costs[index],
        action: actions_taken[index],
        reward: rewards[index]
      };
    });
  };

  // Get recent actions (last 10)
  const getRecentActions = () => {
    const { actions_taken, timestamps, rewards } = simulationData;
    if (!actions_taken || actions_taken.length === 0) return [];
    
    // Get the last 10 actions
    const length = actions_taken.length;
    const start = Math.max(0, length - 10);
    
    return actions_taken.slice(start).map((action, index) => {
      const actualIndex = start + index;
      const time = timestamps[actualIndex] 
        ? new Date(timestamps[actualIndex] * 1000).toLocaleTimeString() 
        : '';
      const reward = rewards[actualIndex] || 0;
      
      return { action, time, reward };
    }).reverse(); // Most recent first
  };

  // Start the simulation
  const startSimulation = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/start-simulation`);
      if (response.data.status === 'success') {
        setSimulationRunning(true);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Failed to start simulation: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Stop the simulation
  const stopSimulation = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/stop-simulation`);
      if (response.data.status === 'success') {
        setSimulationRunning(false);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Failed to stop simulation: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Get current status and simulation data
  const fetchData = async () => {
    try {
      // Get status
      const statusResponse = await axios.get(`${API_URL}/status`);
      setStatusInfo(statusResponse.data);
      setSimulationRunning(statusResponse.data.status === 'running');
      
      // Get simulation data if running
      if (statusResponse.data.status === 'running') {
        const dataResponse = await axios.get(`${API_URL}/simulation-data`);
        setSimulationData(dataResponse.data);
      }
      
      setError(null);
    } catch (err) {
      setError('Failed to fetch data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Initial load and polling for updates
  useEffect(() => {
    // Fetch data immediately on component mount
    fetchData();
    
    // Set up polling interval
    const interval = setInterval(fetchData, 1000);
    
    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  const chartData = formatChartData();
  const recentActions = getRecentActions();

  return (
    <div className="App">
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom component="div" align="center">
          Enterprise AI Monitor
        </Typography>
        
        {/* Status and Controls */}
        <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="body1">
                Status: {simulationRunning ? (
                  <span style={{ color: 'green', fontWeight: 'bold' }}>Running</span>
                ) : (
                  <span style={{ color: 'red' }}>Stopped</span>
                )}
              </Typography>
              <Typography variant="body2">
                Models Loaded: LSTM ({statusInfo.models_loaded.lstm ? 'Yes' : 'No'}), 
                RL Agent ({statusInfo.models_loaded.rl_agent ? 'Yes' : 'No'})
              </Typography>
              {error && (
                <Typography variant="body2" color="error">
                  Error: {error}
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} md={4} sx={{ textAlign: 'right' }}>
              {simulationRunning ? (
                <Button 
                  variant="contained" 
                  color="error" 
                  onClick={stopSimulation}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Stop Simulation'}
                </Button>
              ) : (
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={startSimulation}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Start Simulation'}
                </Button>
              )}
            </Grid>
          </Grid>
        </Paper>
        
        {/* Charts */}
        <Grid container spacing={3}>
          {/* CPU Utilization Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="CPU Utilization (%)" />
              <CardContent>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="cpu" 
                        stroke="#8884d8" 
                        activeDot={{ r: 8 }} 
                        name="CPU (%)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Memory Utilization Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Memory Utilization (%)" />
              <CardContent>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="memory" 
                        stroke="#82ca9d" 
                        activeDot={{ r: 8 }} 
                        name="Memory (%)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Security Threats Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Security Threat Level (%)" />
              <CardContent>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="security" 
                        stroke="#ff8042" 
                        activeDot={{ r: 8 }} 
                        name="Threat Level (%)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Costs Chart */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader title="Operating Costs" />
              <CardContent>
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="cost" 
                        stroke="#ff0000" 
                        activeDot={{ r: 8 }} 
                        name="Cost"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </Grid>
          
          {/* Recent Actions Table */}
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Recent Agent Actions" />
              <CardContent>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Action</TableCell>
                        <TableCell>Reward</TableCell>
                        <TableCell>Impact</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {recentActions.length > 0 ? (
                        recentActions.map((item, index) => (
                          <TableRow key={index}>
                            <TableCell>{item.time}</TableCell>
                            <TableCell>{item.action}</TableCell>
                            <TableCell 
                              style={{ 
                                color: item.reward >= 0 ? 'green' : 'red',
                                fontWeight: 'bold'
                              }}
                            >
                              {item.reward.toFixed(2)}
                            </TableCell>
                            <TableCell>
                              {getActionImpact(item.action)}
                            </TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={4} align="center">
                            No actions recorded yet
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* System Recommendations */}
          <Grid item xs={12}>
            <Card>
              <CardHeader title="System Recommendations" />
              <CardContent>
                <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                  <Typography variant="h6">
                    Current Status Summary
                  </Typography>
                  {chartData.length > 0 ? (
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, bgcolor: getCpuHealthColor(chartData[chartData.length - 1]?.cpu) }}>
                          <Typography variant="subtitle1">
                            CPU: {chartData[chartData.length - 1]?.cpu.toFixed(1)}%
                          </Typography>
                          <Typography variant="body2">
                            {getCpuHealthStatus(chartData[chartData.length - 1]?.cpu)}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, bgcolor: getMemoryHealthColor(chartData[chartData.length - 1]?.memory) }}>
                          <Typography variant="subtitle1">
                            Memory: {chartData[chartData.length - 1]?.memory.toFixed(1)}%
                          </Typography>
                          <Typography variant="body2">
                            {getMemoryHealthStatus(chartData[chartData.length - 1]?.memory)}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, bgcolor: getSecurityHealthColor(chartData[chartData.length - 1]?.security) }}>
                          <Typography variant="subtitle1">
                            Security: {chartData[chartData.length - 1]?.security.toFixed(1)}%
                          </Typography>
                          <Typography variant="body2">
                            {getSecurityHealthStatus(chartData[chartData.length - 1]?.security)}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} md={3}>
                        <Paper sx={{ p: 2, bgcolor: getCostHealthColor(chartData[chartData.length - 1]?.cost) }}>
                          <Typography variant="subtitle1">
                            Cost: ${chartData[chartData.length - 1]?.cost.toFixed(2)}
                          </Typography>
                          <Typography variant="body2">
                            {getCostHealthStatus(chartData[chartData.length - 1]?.cost)}
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  ) : (
                    <Typography variant="body1" sx={{ mt: 2 }}>
                      No data available yet. Start the simulation to see recommendations.
                    </Typography>
                  )}
                </Box>

                {chartData.length > 0 && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="h6">
                      Recommendations:
                    </Typography>
                    <ul>
                      {getSystemRecommendations(chartData[chartData.length - 1]).map((rec, index) => (
                        <li key={index}>
                          <Typography variant="body1">{rec}</Typography>
                        </li>
                      ))}
                    </ul>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

// Helper functions for the UI

// Get action impact description
function getActionImpact(action) {
  switch (action) {
    case 'INCREASE_RESOURCES':
      return 'CPU & Memory ↓, Cost ↑';
    case 'DECREASE_RESOURCES':
      return 'CPU & Memory ↑, Cost ↓';
    case 'MAINTAIN_RESOURCES':
      return 'No significant changes';
    case 'ENHANCE_SECURITY':
      return 'Security ↑, Cost ↑';
    case 'OPTIMIZE_COST':
      return 'Cost ↓, Performance may be affected';
    default:
      return 'Unknown impact';
  }
}

// Color functions for health indicators
function getCpuHealthColor(cpu) {
  if (cpu === undefined) return '#f5f5f5';
  if (cpu < 30) return '#c8e6c9'; // Light green (low utilization)
  if (cpu >= 30 && cpu <= 80) return '#c8e6c9'; // Light green (good range)
  if (cpu > 80 && cpu <= 90) return '#fff9c4'; // Light yellow (warning)
  return '#ffcdd2'; // Light red (critical)
}

function getMemoryHealthColor(memory) {
  if (memory === undefined) return '#f5f5f5';
  if (memory < 30) return '#c8e6c9'; // Light green (low utilization)
  if (memory >= 30 && memory <= 80) return '#c8e6c9'; // Light green (good range)
  if (memory > 80 && memory <= 90) return '#fff9c4'; // Light yellow (warning)
  return '#ffcdd2'; // Light red (critical)
}

function getSecurityHealthColor(security) {
  if (security === undefined) return '#f5f5f5';
  if (security < 5) return '#c8e6c9'; // Light green (low threat)
  if (security >= 5 && security <= 15) return '#fff9c4'; // Light yellow (moderate threat)
  return '#ffcdd2'; // Light red (high threat)
}

function getCostHealthColor(cost) {
  if (cost === undefined) return '#f5f5f5';
  if (cost < 30) return '#c8e6c9'; // Light green (low cost)
  if (cost >= 30 && cost <= 60) return '#c8e6c9'; // Light green (moderate cost)
  if (cost > 60 && cost <= 80) return '#fff9c4'; // Light yellow (high cost)
  return '#ffcdd2'; // Light red (very high cost)
}

// Status text functions
function getCpuHealthStatus(cpu) {
  if (cpu === undefined) return 'No data';
  if (cpu < 30) return 'Underutilized';
  if (cpu >= 30 && cpu <= 80) return 'Optimal';
  if (cpu > 80 && cpu <= 90) return 'Warning: High utilization';
  return 'Critical: Over-utilized';
}

function getMemoryHealthStatus(memory) {
  if (memory === undefined) return 'No data';
  if (memory < 30) return 'Underutilized';
  if (memory >= 30 && memory <= 80) return 'Optimal';
  if (memory > 80 && memory <= 90) return 'Warning: High utilization';
  return 'Critical: Over-utilized';
}

function getSecurityHealthStatus(security) {
  if (security === undefined) return 'No data';
  if (security < 5) return 'Low threat level';
  if (security >= 5 && security <= 15) return 'Moderate threat level';
  return 'High threat level';
}

function getCostHealthStatus(cost) {
  if (cost === undefined) return 'No data';
  if (cost < 30) return 'Low cost';
  if (cost >= 30 && cost <= 60) return 'Moderate cost';
  if (cost > 60 && cost <= 80) return 'High cost';
  return 'Very high cost';
}

// Get system recommendations based on current state
function getSystemRecommendations(currentState) {
  if (!currentState) return [];
  
  const recommendations = [];
  
  // CPU recommendations
  if (currentState.cpu > 90) {
    recommendations.push('CRITICAL: CPU is over-utilized. Immediate resource scaling recommended.');
  } else if (currentState.cpu > 80) {
    recommendations.push('CPU utilization is high. Consider increasing resources if this trend continues.');
  } else if (currentState.cpu < 30) {
    recommendations.push('CPU is underutilized. Consider resource optimization to reduce costs.');
  }
  
  // Memory recommendations
  if (currentState.memory > 90) {
    recommendations.push('CRITICAL: Memory is over-utilized. Immediate resource scaling recommended.');
  } else if (currentState.memory > 80) {
    recommendations.push('Memory utilization is high. Consider increasing resources if this trend continues.');
  } else if (currentState.memory < 30) {
    recommendations.push('Memory is underutilized. Consider resource optimization to reduce costs.');
  }
  
  // Security recommendations
  if (currentState.security > 15) {
    recommendations.push('CRITICAL: High security threat detected. Immediate action required.');
  } else if (currentState.security > 5) {
    recommendations.push('Moderate security threats detected. Consider enhancing security measures.');
  }
  
  // Cost recommendations
  if (currentState.cost > 80) {
    recommendations.push('Cost is very high. Review resource allocation for optimization opportunities.');
  } else if (currentState.cost > 60) {
    recommendations.push('Cost is higher than optimal. Consider cost optimization strategies.');
  }
  
  // If everything is optimal
  if (recommendations.length === 0) {
    recommendations.push('All systems are operating at optimal levels. No immediate action required.');
  }
  
  return recommendations;
}

export default App;