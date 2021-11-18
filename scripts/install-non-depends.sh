#!/usr/bin/env bash

SUDO=""
if [[ "$(whoami)" == "root" ]];
then
    SUDO=""
else
    SUDO="sudo"
fi

curl -sSL https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-linux-arm64 > bazel
$SUDO mv bazel /bin/bazel
$SUDO chmod +x /bin/bazel
