#! /usr/bin/env bash  

# Install Python dependencies directly into system Python (no virtual environment)
sudo uv pip install --system -e .[dev]

# Install Node.js dependencies
pushd inspector
npm install
popd
