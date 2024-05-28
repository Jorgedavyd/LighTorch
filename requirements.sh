rm requirements.txt
pipreqs lightorch/ --savepath=requirements.in
pip-compile
rm requirements.in
