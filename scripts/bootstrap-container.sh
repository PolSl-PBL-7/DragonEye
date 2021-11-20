#!/usr/bin/env bash

# first, determine if user running this script is a root
# if not, then set sudo for apt commands
if [[ "$(whoami)" == "root" ]]
then
    SUDO=""
else
    SUDO="sudo"
fi

# update repositories
$SUDO apt-get update -y

# install all dependencies
cat debian.depends | $SUDO xargs apt-get install -y

# install dependencies that are not found inside 
# any repository by default
sh -c ./scripts/install-non-depends.sh

python3 ./scripts/bootstrap.py

python3 -m pip install -r ./requirements.txt
