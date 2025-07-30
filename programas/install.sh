#!/bin/bash
sudo apt-get update
sudo apt-get install python3 python3-venv
sudo apt-get install python3-tk
python3 -m venv case_environment
source /home/zozimo/sase_machine_learning/programas/case_enviroment/bin/activate
pip3 install torch torchvision scipy matplotlib opencv-python PyQt6 ultralytics
