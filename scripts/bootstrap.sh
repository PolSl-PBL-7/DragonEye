#!/usr/bin/env bash

# determine the operating system
OS=$(uname -s)

# install dependencies and OS specific features
if [[ "$OS" == "Darwin" ]]
then
    source ~/miniforge3/bin/activate

    cat macos.depends | xargs brew install

elif [[ "$OS" == "Linux" ]]
then
    # determine os type
    OS_TYPE=$(cat /etc/issue.net)

    if [[ "$OS_TYPE" == *"Debian"* || "$OS_TYPE" == *"Ubuntu"* ]]
    then
        sudo apt-get update -y
        cat debian.depends | sudo xargs apt-get install -y --no-install-recommends
        sh -c ./scripts/install-non-depends.sh
    else
        echo "OS not supported"
        exit 255
    fi

    # create venv for python
    python3 -m venv venv

    # source the venv configuration
    source ./venv/bin/activate
fi

# run bootstrapper
python3 ./scripts/bootstrap.py

# install dependencies requirements.txt
python3 -m pip install -r ./requirements.txt
