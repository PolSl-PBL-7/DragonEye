#!/usr/bin/env sh

if [[ -z "${PIP_EXTRA_INDEX_URL}" ]]; then
    export PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/
fi

pipenv uninstall tensorflow && \
    pip install tensorflow-aarch64==2.7.* -f $PIP_EXTRA_INDEX_URL
