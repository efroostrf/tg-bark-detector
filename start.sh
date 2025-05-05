#!/bin/bash

# Start the bark detector in the background
python main.py &
MAIN_PID=$!

# Start the FastAPI server in the background
python api.py &
API_PID=$!

# Function to cleanup on exit
cleanup() {
  echo "Stopping services..."
  kill -TERM $MAIN_PID 2>/dev/null
  kill -TERM $API_PID 2>/dev/null
  wait
  echo "All services stopped"
  exit 0
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n

# If we get here, one of the processes exited unexpectedly
echo "One of the processes exited unexpectedly"
cleanup 