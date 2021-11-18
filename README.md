# DragonEye

## Bootstraping developement environment

If you want to have your environment set with default libraries hassle free and easy, simply run the following command from root of the project:

```bash
./scripts/bootstrap.sh
```

or on Windows:

```powershell
./scripts/bootstrap.ps1
```

This will download correct version of TensorFlow (by default it is just `tensorflow, but it will install `tensorflow-macos` and `tensorflow-metal` for MacOS, and `tensorflow-aarch64` built by [Linaro](https://snapshots.linaro.org/ldcg/python-cache/) on *aarch64/Linux*).

In addition to the TensorFlow, on *aarch64/MacOS* this will attempt to install [minifroge](https://github.com/conda-forge/miniforge) - a minimal, community-led collection of recipes, build infrastructure and distributions for the conda package manager. To avoid installing `miniforge` (if you already have it installed), pass the `--skip` parameter to the bootstrap script:

```bash
./scripts/bootstrap.sh
```

You can install just miniforge, by running `./scripts/install-miniforge.sh`, create new virtual environment with conda, and install `requirements.txt` and TensorFlow by hand.

## Windows scripts

In the `scripts` directory, `install.ps1` and `profile.ps1` scripts can be found.
Use the following command to get additionall info about installation script:

```powershell
install.ps1  -help
```

If you used `install.ps1` you can load `profile.ps1` by issuing command:

```powershell
. profile.ps1
```

If you want to export path to use it inside PyCharm or VSCode, then run:
```powershell
profile.ps1 -envfile "windows.env" -export
```

# Saving dependencies

In order to create proper `requirements.txt` file, run:

```sh
make freeze
```

Or if you are using Conda:

```sh
make conda-freeze
```
