#!/usr/bin/env bash

OS=$(uname -s)

if [[ "$OS" == "Darwin" ]];
then
    source ~/miniforge3/bin/activate

    cat macos.depends | xargs brew install

elif [[ "$OS" == "Linux" ]];
then
    OS_TYPE=$(cat /etc/issue.net)

    if [[ "$OS_TYPE" == *"Debian"* || "$OS_TYPE" == *"Ubuntu"* ]];
    then
        sudo apt-get update -y
        cat debian.depends | sudo xargs apt-get install -y
        sh -c ./scripts/install-non-depends.sh
    fi

    python3 -m venv venv

    source ./venv/bin/activate
fi

python3 ./scripts/bootstrap.py

python3 -m pip install -r ./requirements.txt
