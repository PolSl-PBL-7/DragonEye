#!/usr/bin/env sh

pipenv uninstall tensorflow && pipenv install https://github.com/KumaTea/tensorflow-aarch64/releases/download/v2.6/tensorflow-2.6.0-cp39-cp39-linux_aarch64.whl
