# PBL main repository

This repository contains main code of the Project Based Learning 
programme repositry.

### Windows scripts

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
