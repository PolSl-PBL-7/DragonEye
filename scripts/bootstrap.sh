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

python3 -m pip --version || sudo apt-get install -y python3-pip

if [[ "$(python3 --version)" != *"3.9."* ]]
then
    sudo apt-get install -y python3.9
    PYTHON3=$(which python3)
    PYTHON39=$(which python3.9)
    mv $PYTHON3 $PYTHON3.bak
    mv $PYTHON39 $PYTHON3
fi

# install all dependencies
cat debian.depends | $SUDO xargs apt-get install -y --no-install-recommends

# install dependencies that are not found inside 
# any repository by default
sh -c ./scripts/install-non-depends.sh

python3 ./scripts/bootstrap.py

python3 -m pip install -r ./requirements.txt
