## DragonEye

1. [Prerequisites](#prerequisites)
1. [Bootstraping developement environment](#bootstraping-developement-environment)
1. [Updating requirements.txt](#updating-requirements.txt)
1. [Static code analysis](#static-code-analysis)
1. [Running tests](#running-tests)

## Prerequisites

### Linux

1. One of officially supported distros:
    - Debian Bullseye;
    - Ubuntu 20.04 LTS;
    - Pop!_OS 20.04 LTS;
    - WSL Ubuntu 20.04 LTS;
1. Installed Python, version 3.9;
1. Installed Docker;
1. Installed GNU Make;

## Bootstraping developement environment

The general flow of bootstraping looks like this:

1. Installing third party dependecies using package managers (*apt-get*) and install script;
1. Installing special dependencies for specific systems from `bootstrap.toml`;
1. Installing dependencies from `requirements.txt`;

If you want to have your environment set with default libraries hassle free and easy, simply run the following command from root of the project:

```bash
./scripts/bootstrap.sh
```

---

## Using WSL with CUDA support

TODO: make install-cuda, driver 510 or grater

## Updating `requirements.txt`

As we blacklist few dependencies from appearing in `requirements.txt`, you need to use our custom script to update it.

In order to create proper `requirements.txt` file, run:

```sh
make freeze
```

## Static code analysis

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
