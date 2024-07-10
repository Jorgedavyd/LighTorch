#!/bin/bash
black .
git config --global user.name 'Jorgedavyd'
git config --global user.email 'jorged.encyso@gmail.com'
git diff --exit-code || (git add . && git commit -m "Automatically formatted with black" && git push)

