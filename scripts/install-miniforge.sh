#!/usr/bin/env bash

# config
MINIFORGE_3_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

function install-miniforge {
    # download miniforge install script
    curl -sSL $MINIFORGE_3_URL > tmp.sh

    # set script as executable
    chmod u+x ./tmp.sh

    # run the script
    sh ./tmp.sh

    # remove the script
    rm -f ./tmp.sh
}

install-miniforge
