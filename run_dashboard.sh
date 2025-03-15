#!/bin/bash
# Run the dashboard with the correct path settings

# Get the absolute path to the project directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PROJECT_ROOT}"

# Run the dashboard
python "${PROJECT_ROOT}/src/dashboard/app.py" "$@"

# Note: You can pass additional arguments to this script, like:
# ./run_dashboard.sh --debug 