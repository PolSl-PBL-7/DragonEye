#!/usr/bin/env bash

# configuration
BAZEL_VERSION=4.2.1

# first, determine if user running this script is a root
# if not, then set sudo for apt commands
if [[ "$(whoami)" == "root" ]]
then
    SUDO=""
else
    SUDO="sudo"
fi

# determine architecture
# uname -p sometimes returns `unknown` in docker, thus using python
# and heredoc
ARCH=$(python3 <<EOF
import platform as p
print(p.machine())
EOF
)


function install_bazel {
    if [[ "$ARCH" == "aarch64" ]]
    then
        BAZEL_URI=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-arm64
    else
        BAZEL_URI=https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64
    fi

    # download bazel binary
    curl -sSL $BAZEL_URI > bazel

    # move it to the target location and set executble bits
    $SUDO mv bazel /bin/bazel
    $SUDO chmod +x /bin/bazel

    bazel
}

install_bazel
