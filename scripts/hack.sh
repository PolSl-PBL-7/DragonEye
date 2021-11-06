#!/usr/bin/env bash

ARCH=$(uname -m)
SYSTEM=$(uname -s)

if [[ $SYSTEM != "Darwin" ]]; then
    echo "System not supported!"
    exit 1
fi

if [[ $ARCH == "arm64" ]]; then
    MINIFORGE_3_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
elif [[ $ARCH == "x86_64" ]]; then
    MINIFORGE_3_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
else
    echo "System not supported!"
    exit 1
fi

function install {
    curl -sSL $MINIFORGE_3_URL > tmp.sh
    chmod u+x ./tmp.sh
    sh ./tmp.sh

    source ~/miniforge3/bin/activate

    conda install -c apple tensorflow-deps --force-reinstall
    python3 -m pip install tensorflow-macos
    python3 -m pip install tensorflow-metal
    python3 -m pip install pipenv
}

option="${1}"
case $option in 
    "install")
        install
        exit 0
    ;;
esac

alias py="python3"

source ~/miniforge3/bin/activate

export PYTHONPATH=~/miniforge3/pkgs:/Users/shanduur/miniforge3/pkgs:/Users/shanduur/miniforge3/lib/python3.9:/Users/shanduur/miniforge3/lib/python3.9/lib-dynload:/Users/shanduur/miniforge3/lib/python3.9/site-packages

py -m pipenv shell
