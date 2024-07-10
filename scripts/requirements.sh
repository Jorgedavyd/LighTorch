#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install pytest black pipreqs pip-tools
pipreqs lightorch --savepath=./requirements.in
sed -i 's/==.*$//' requirements.in
sort requirements.in | uniq > requirements_unique.in
pip-compile ./requirements_unique.in
rm -f *.in
pip install -r requirements_unique.txt
rm -rf requirements_unique.txt
