## DragonEye

1. [Prerequisites](#prerequisites)
1. [Bootstraping developement environment](#bootstraping-developement-environment)
1. [Using container with CUDA support](#using-container-with-cuda-support)
1. [Updating requirements.txt](#updating-requirements.txt)
1. [Static code analysis](#static-code-analysis)
1. [Running tests](#running-tests)

## Prerequisites

### Linux

1. One of officially supported distros:
    - Debian Bullseye;
    - Ubuntu 20.04 LTS;
    - Pop!_OS 20.04 LTS;
1. Installed Python, version 3.9;
1. Installed Docker;
1. Installed GNU Make;

## Windows

1. Installed Python, version 3.9;
1. Installed WSL 2.0;
1. Installed Docker;
1. Installed GNU Make for Windows;

Optional:
1. Installed proper NVidia GeForce Driver;
1. Installed CUDA Toolkit and cuDNN (can be done using [Windows scripts](#windows-scripts));
1. Installed nvidia support for Docker;
1. Installed GNU Make;

### MacOS

1. Installed Python, version 3.9;
1. Installed conda (can be done using [MacOS scripts](#macos-scripts));
1. Installed homebrew;

## Bootstraping developement environment

The general flow of bootstraping looks like this:

1. Installing third party dependecies using package managers (*homebrew* for MacOS, *apt-get* for Debian/Ubuntu) or install script (*Windows* and *Linux*)
1. Installing special dependencies for specific systems from `bootstrap.toml`;
1. Installing dependencies from `requirements.txt`;

If you want to have your environment set with default libraries hassle free and easy, simply run the following command from root of the project:

```bash
./scripts/bootstrap.sh
```

or on Windows:

```powershell
./scripts/bootstrap.ps1
```

This will download correct version of TensorFlow (by default it is just `tensorflow`, but it will install `tensorflow-macos` and `tensorflow-metal` for MacOS, and `tensorflow-aarch64` built by [Linaro](https://snapshots.linaro.org/ldcg/python-cache/) on *aarch64/Linux*).

---

**! WARNING !**

Currently support for MacOS is experimental. As for today (11/20/2021) there is no TensorFlow `2.7.*` compiled for MacOS, so the app won't run. Additionally, some dependencies from `requirements.txt` won't install correctly on Apple Silicon (*aarch64/Darwin*).

---

### MacOS Scripts

You can install miniforge, by running `./scripts/install-miniforge.sh`, create new virtual environment with conda, and install `requirements.txt` and TensorFlow by hand.

### Windows scripts

In the `scripts` directory, `install.ps1` and `profile.ps1` scripts can be found.
Use the following command to get additionall info about installation script:

```powershell
install.ps1  -help
```

If you used `install.ps1` you can load `profile.ps1` by issuing command:

```powershell
. .\scripts\profile.ps1
```

If you want to export path to use it inside PyCharm or VSCode, then run:
```powershell
.\scripts\profile.ps1 -envfile "windows.env" -export
```

## Using container with CUDA support

If you want to run the *devcontainer* with support for CUDA acceleration, make sure you have all necessary setup done on your PC:

- **For Windows Users**: follow [WSL user guide by NVidia](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- **For Linux Users**: follow [NVidia container toolkit install guide by NVidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

After that, uncomment `"runArgs": [ "--gpus=all" ],` in `devcontainer/devcontainer.json`.

Reopen the folder in devcontainer, and run the following command

```sh
make install-cuda
```

If you can see name of your GPU in the table at the end of the output, everything should be working.

## Updating `requirements.txt`

As we blacklist few dependencies from appearing in `requirements.txt`, you need to use our custom script to update it.

In order to create proper `requirements.txt` file, run:

```sh
make freeze
```

Or if you are using Conda:

```sh
make conda-freeze
```

If you are using conda, only packages installed by the `pip` will be listed, thus be aware that some uninstalled dependecies might be lost from `requirements.txt`.

## Static code analysis

### mypy

We are running automatic *mypy* tests to verify if there are any potential typing problems. To run *mypy* type the following command in the terminal:

```sh
make mypy
```

### flake8

Our code needs to pass the quality check, that ensures that standards are preserved. To achieve this, we are using *flake8*. To run *flake8* verification type the following command in the terminal

```sh
make flake
```

### autopep8

Some issues found by *flake8* can be fixed by *autopep8*. To run *autopep8* type the following command in the terminal:

```sh
make autopep
```

## Running tests

### Tests on your environment

To run tests on your local environment type the following command in the terminal:

```sh
make test
```

### Testing local version of CI

To run tests on isolated environment inside Docker type the following command in the terminal:

```sh
make local-ci
```

### Automated tests in cloud

Every pull request is tested on the *Drone.CI*. Those tests should be identical to the local version of CI mentioned above, but might take lot longer to run. If you think all those test can run faster on your own machine, simply run `make local-ci` before opening pull request, to verify all potential issues.
