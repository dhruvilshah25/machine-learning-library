#!/bin/bash

pip install pandas
pip install tqdm
pip install matplotlib
pip install numpy
pip install scipy
python3 driver.py primal
python3 driver.py dual
python3 driver.py gk
