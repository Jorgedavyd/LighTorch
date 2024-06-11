#!/bin/bash

pipreqs lightorch --savepath=requirements.in

pip-compile requirements.in 

rm -f requirements.in