#!/bin/bash

# This script monitors the CPU and GPU usage of a specified ROS2 node.
# Usage: ./monitor.sh <node_name>
# Example: ./monitor.sh lucas_kanade_node

if [ -z "$1" ]; then
    echo "Usage: $0 <node_name>"
    exit 1
fi

NODE_NAME=$1

# Mapping of node names to their install paths
declare -A NODE_PATHS=(
    ["lucas_kanade_node"]="/home/docker/OpticalFlowSlider/ros2_ws/install/optical_flow/lib/optical_flow"
    ["lucas_kanade_light_node"]="/home/docker/OpticalFlowSlider/ros2_ws/install/optical_flow/lib/optical_flow"
    ["lucas_kanade_accurate_node"]="/home/docker/OpticalFlowSlider/ros2_ws/install/optical_flow/lib/optical_flow"
    ["raft_direct_node"]="/home/docker/OpticalFlowSlider/ros2_ws/install/optical_flow/lib/optical_flow"
    ["raft_small_node"]="/home/docker/OpticalFlowSlider/ros2_ws/install/optical_flow/lib/optical_flow"
    ["pwc_sub_node"]="/home/docker/ROS2_liteflownet/ros2_ws/install/pwc_net/lib/pwc_net"
    ["lfn3_sub_node"]="/home/docker/ROS2_liteflownet/ros2_ws/install/liteflownet3/lib/liteflownet3"
)

# Check if the node name is in the mapping
if [ -z "${NODE_PATHS[$NODE_NAME]}" ]; then
    echo "Error: Unknown node name '$NODE_NAME'. Supported nodes: ${!NODE_PATHS[@]}"
    exit 1
fi

INSTALL_PATH="${NODE_PATHS[$NODE_NAME]}"

echo "Waiting for $NODE_NAME to start..."

# Wait for the node to start
while true; do
    # Find the PID, ensuring we pick the actual node process (not the launcher)
    PID=$(pgrep -f $INSTALL_PATH/$NODE_NAME | grep -v "ros2 run" | head -n 1)
    if [ -n "$PID" ]; then
        break
    fi
    echo "Process not found. Retrying in 1 second..."
    sleep 1
done

echo "Monitoring PID $PID for $NODE_NAME"

# Function to check if the process is still running
check_process() {
    ps -p $PID > /dev/null
    return $?
}

# Start CPU monitoring in the background
pidstat -p $PID 1 > cpu_usage_$NODE_NAME.log &

# Store the PID of pidstat
PIDSTAT_PID=$!

# Start GPU monitoring (optional, if GPU is used)
if command -v nvidia-smi &> /dev/null; then
    while true; do
        if ! check_process; then
            echo "Node $NODE_NAME (PID $PID) has stopped. Exiting GPU monitoring."
            break
        fi
        nvidia-smi pmon | grep -E "\b$PID\b" >> gpu_usage_$NODE_NAME.log
        sleep 1
    done &
    GPU_MONITOR_PID=$!
else
    echo "nvidia-smi not found, skipping GPU monitoring."
fi

# Monitor the process and stop when it exits
while check_process; do
    sleep 1
done

echo "Node $NODE_NAME (PID $PID) has stopped. Stopping monitoring."

# Kill the background monitoring processes
if [ -n "$PIDSTAT_PID" ]; then
    kill $PIDSTAT_PID 2>/dev/null
fi
if [ -n "$GPU_MONITOR_PID" ]; then
    kill $GPU_MONITOR_PID 2>/dev/null
fi

exit 0