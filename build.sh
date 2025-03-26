#!/bin/bash
set -e

# Install system dependencies
apt-get update
apt-get install -y build-essential python3-dev

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt 