#!/bin/bash

# Stop the script on any error
set -e

# Check if conda command exists
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Installing Miniconda..."
    
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
    
    # Install Miniconda silently
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    
    # Initialize Conda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    
    # Clean up installer
    rm Miniconda3-latest-Linux-x86_64.sh
    
    echo "Miniconda installed successfully."
fi

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)

if [ -z "${CONDA_BASE}" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

PATH="${CONDA_BASE}/bin/":$PATH
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_NAME="three-gen-mining"
# Check if the environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists."
    echo "[INFO] Running cleanup script to remove the environment."
    
    # Run cleanup_env.sh from the same directory as this script
    ./cleanup_env.sh
else
    echo "[INFO] Environment '${ENV_NAME}' does not exist. Proceeding to create it."
fi

# Create environment and activate it
conda env create -f conda_env_mining.yml -y
conda activate three-gen-mining
conda info --env

CUDA_HOME=${CONDA_PREFIX}
echo -e "\n\n[INFO] Installing diff-gaussian-rasterization package\n"
mkdir -p ./extras/diff_gaussian_rasterization/third_party
git clone --branch 0.9.9.0 https://github.com/g-truc/glm.git ./extras/diff_gaussian_rasterization/third_party/glm
pip install -q ./extras/diff_gaussian_rasterization

echo -e "\n\n[INFO] Installing simple-knn package\n"
pip install -q ./extras/simple-knn

echo -e "\n\n[INFO] Installing MVDream package\n"
pip install -q ./extras/MVDream

echo -e "\n\n[INFO] Installing ImageDream package\n"
pip install -q ./extras/ImageDream

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Check if profiling is enabled
PROFILE_FLAG=""
if [[ "$1" == "--profile" ]]; then
    PROFILE_FLAG="--profile"
    echo "[INFO] Profiling is enabled."
else
    echo "[INFO] Profiling is disabled."
fi

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 8093 ${PROFILE_FLAG}'
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."