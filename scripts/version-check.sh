#!/usr/bin/env sh

REPO=https://raw.githubusercontent.com/PolSl-PBL-7/DragonEye/

# STRING=__version__
STRING=max-complexity\ =\ 10
FILE=.flake8
BRANCH=docs-fill-readme

MAIN_BRANCH=$(curl -sSL $REPO/main/$FILE)
echo $MAIN_BRANCH
UPDATE_BRANCH=$(curl -sSL $REPO/$BRANCH/$FILE)
echo $UPDATE_BRANCH

MAIN_VERSION=$(echo $MAIN_BRANCH | grep "$STRING")
echo $MAIN_VERSION
UPDATE_VERSION=$(echo $UPDATE_BRANCH | grep "$STRING")
echo $UPDATE_VERSION

if [[ "$MAIN_VERSION" == "$UPDATE_VERSION" ]];
then
    exit 2
fi