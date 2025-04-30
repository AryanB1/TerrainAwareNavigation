#!/bin/bash
set -e

echo "[*] Starting full ROS + ML environment setup..."

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$PROJECT_ROOT/ros2_ws"
VENV_DIR="$PROJECT_ROOT/venv"

# -------------------------
# System Prep
# -------------------------
echo "[*] Updating system and installing dev packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl build-essential cmake ninja-build python3-venv python3-pip

# -------------------------
# CUDA Setup
# -------------------------
echo "[*] Installing CUDA Toolkit 12.3..."
[ -f "$PROJECT_ROOT/cuda-keyring_1.1-1_all.deb" ] || wget -O "$PROJECT_ROOT/cuda-keyring_1.1-1_all.deb" https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# -------------------------
# ROS 2 Humble
# -------------------------
echo "[*] Installing ROS 2 Humble Desktop..."
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo tee /etc/apt/trusted.gpg.d/ros.asc > /dev/null
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update

sudo apt install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update

echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

# -------------------------
# OpenCV + Vision
# -------------------------
echo "[*] Installing OpenCV and ROS bindings..."
sudo apt install -y libopencv-dev python3-opencv ros-humble-vision-opencv

# -------------------------
# Python Virtual Environment
# -------------------------
echo "[*] Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[*] Installing PyTorch (CUDA-enabled) and other Python tools..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib jupyter scikit-learn

echo "source $WORKSPACE/install/setup.bash" >> ~/.bashrc
echo "source $VENV_DIR/bin/activate" >> ~/.bashrc

# -------------------------
# Done
# -------------------------
echo "Setup complete:"
echo "run: source ~/.bashrc and colcon build"
