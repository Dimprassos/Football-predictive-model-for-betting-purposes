#!/bin/bash

# Define colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}Setting up the Football Prediction Project${NC}"
echo -e "${CYAN}==========================================${NC}"

echo -e "\nChecking for existing virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${YELLOW}Creating virtual environment venv without pip to avoid hangs...${NC}"
    python3 -m venv venv --without-pip
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

echo "Activating virtual environment..."
source venv/bin/activate

if ! python3 -m pip --version >/dev/null 2>&1; then
    echo -e "${YELLOW}pip is missing. Installing via get-pip.py...${NC}"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
fi

echo -e "\nUpgrading pip to the latest version..."
python3 -m pip install --upgrade pip

echo -e "\nInstalling dependencies..."
# Χρησιμοποιούμε forward slashes (/) για τα paths στο Linux
python3 -m pip install -r requirments/requirments.txt

echo -e "\n${CYAN}==========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo "To activate the virtual environment and run the model, type:"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}python3 main.py${NC}"
echo -e "${CYAN}==========================================${NC}"