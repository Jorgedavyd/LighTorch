#!/bin/bash
rm -rf requirements.txt

pipreqs lightorch --savepath=requirements.in

pip-compile requirements.in 

rm -f requirements.in
