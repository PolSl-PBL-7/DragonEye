#!/usr/bin/env sh

pipenv uninstall tensorflow && pipenv install tensorflow-aarch64 --skip-lock
