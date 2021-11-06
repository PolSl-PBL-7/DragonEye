# PBL main repository

This repository contains main code of the Project Based Learning 
programme repositry.

## Installation

TODO

### MacOS Hack

If you are using MacOS, then probably installation of TensorFlow will fail (especially on ARM Macs). To address this issue, you can use environment inside Docker or use the hack we made.

To install TensorFlow run:

```sh
./scripts/hack.sh install
```

You will be prompted for some configuration (it will install [miniforge](https://github.com/conda-forge/miniforge) and necessary packages).

After succesfull installation, run:

```sh
source ./scripts/hack.sh
```

This command will launch a Pipenv with additional conda-forge configuration including `tensorflow-macos` and `tensorflow-metal`.

## Usage

TODO
