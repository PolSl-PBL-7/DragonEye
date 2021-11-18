# Template Repository

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
