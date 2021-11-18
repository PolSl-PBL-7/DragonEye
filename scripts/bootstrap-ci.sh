#!/usr/bin/env bash

apt-get update -y
cat debian.depends | apt-get install -y
sh -c ./scripts/install-non-depends.sh

python3 ./scripts/bootstrap.py

python3 -m pip install -r ./requirements.txt
