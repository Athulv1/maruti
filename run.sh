#!/bin/bash

# Face Detection Application Runner
# This script activates the virtual environment and runs the application

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo "Then: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if packages are installed
python -c "import face_recognition" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting Face Detection Application..."
python face_detection_app.py

# Deactivate virtual environment on exit
deactivate
