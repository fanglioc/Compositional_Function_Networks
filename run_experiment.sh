#!/bin/bash

# Get the base name of the experiment file
experiment_name=$(basename "$1" .py)

# Create the log directory if it doesn't exist
mkdir -p experiment_records/training_logs

# Remove old log file to ensure a fresh log for each run
rm -f "experiment_records/training_logs/${experiment_name}.log"

echo "Running experiment: $1"
echo "Logging output to: experiment_records/training_logs/${experiment_name}.log"

# Run the experiment and save the output to a log file
python3 "$1" | tee "experiment_records/training_logs/${experiment_name}.log"