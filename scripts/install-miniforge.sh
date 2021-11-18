#!/usr/bin/env bash

MINIFORGE_3_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

function install-miniforge {
    curl -sSL $MINIFORGE_3_URL > tmp.sh
    chmod u+x ./tmp.sh
    sh ./tmp.sh
    rm -f ./tmp.sh
}

install-miniforge
