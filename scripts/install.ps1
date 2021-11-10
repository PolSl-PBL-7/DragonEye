param (
    [string] $tmp = "C:\tmp",
    [string] $7zip = "C:\Program Files\7-Zip",
    [switch] $dry_run = $false,
    [switch] $explain = $false,
    [switch] $install_all = $false,
    [switch] $install_cuda = $false,
    [switch] $install_cudnn = $false,
    [switch] $install_ffmpeg = $false,
    [switch] $install_wsls1 = $false,
    [switch] $install_wsls2 = $false,
    [switch] $install_wsls3 = $false,
    [switch] $test = $false,
    [switch] $help = $false
)

$ErrorActionPreference = "Stop" # stop execution on error
$ProgressPreference = "SilentlyContinue"

if ($help) {
    Write-Host 'Installer script that enables quick deployment of some necessary components of the Repository.'
    Write-Host 'Instead of using "windows-depends" most of the components are being installed with this scirpt.'
    Write-Host ''
    Write-Host 'WARNING! Script does not perform installation of GeForce Game Ready Driver'
    Write-Host ''
    Write-Host 'Switches:'
    Write-Host '  -dry_run - perform dry run, without making any non-ephemeral changes, enables "explain"' 
    Write-Host '  -explain - additional informations will be displayed at crucial points, script execution will be paused'
    Write-Host '  -install_all - installation of all components excluding WSL'
    Write-Host '  -install_cuda - installation of CUDA Toolkit'
    Write-Host '  -install_cudnn - installation of cuDNN software components for CUDA'
    Write-Host '  -install_ffmpeg - installation of ffmpeg software'
    Write-Host '  -install_wsls1 - stage 1 of installation of components for WSL - enabling required Windows features'
    Write-Host '  -install_wsls2 - stage 2 of installation of components for WSL - WSL2 update installation'
    Write-Host '  -install_wsls3 - stage 3 of installation of components for WSL - installing Ubuntu 20.04 LTS'
    Write-Host ''
    Write-Host 'Parameters:'
    Write-Host '  -tmp [PATH] - points to the place where temporary directory will be created'
    Write-Host '  -7zip [PATH] - points to the place where 7zip is installed'
    Write-Host ''

    exit 0
}

if ($dry_run) {
    $explain = $true
}

if (Test-Path -Path "$7zip\7z.exe") {
    $env:PATH += ";$7zip"
} else {
    Write-Error "7zip not found in $7zip. Install 7zip or provide path to it using -7zip parameter."
}

function explain {
    param (
        [string] $m
    )
    if ($explain) {
        Write-Host $m
        waitForPress
    }
}

function waitForPress {
    Write-Host 'Press any key to continue...'
    $null = $Host.UI.RawUI.ReadKey('IncludeKeyDown')
}

function IsAdmin {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (!$currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "This script must be run as Administrator"
        exit 1
    }
}

function Prepare {
    explain -m "Creating $tmp directory"
    New-Item "$tmp" -ItemType "directory" -Force | Out-Null
}

function Cleanup {
    explain -m "Removing C:\tmp directory"
    Remove-Item $tmp -Recurse
}

function Install-WslStep1 {
    param (
        [string] $tmp = "C:\tmp"
    )
    
    IsAdmin

    explain -m "Enabling feature - WSL"
    if (!$dry_run) {
        dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    }
    
    explain -m "Enable feature - Virtual Machine Platform"
    if (!$dry_run) {
        dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    }

    Write-Host "Rebooting..."
    if (!$dry_run) {
        Restart-Computer
    }
}

function Install-WslStep2 {
    param (
        [string] $tmp = "C:\tmp"
    )
    
    IsAdmin

    Write-Host "Downloading WSL Update..."
    $wsl_update_uri = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
    if (!$dry_run) {
        Invoke-WebRequest $wsl_update_uri -OutFile "$tmp\wsl_update_x64.msi"
    }
    Write-Host "OK"

    Write-Host "Rebooting..."
    if (!$dry_run) {
        Restart-Computer
    }
}

function Install-WslStep3 {
    param (
        [string] $tmp = "C:\tmp"
    )

    IsAdmin
    wsl --set-default-version 2

    Write-Host "Downloading Ubuntu 20.04 LTS..."
    $wsl_ubuntu_2004 = "https://aka.ms/wslubuntu2004"
    Invoke-WebRequest $wsl_ubuntu_2004 -OutFile "$tmp\ubuntu2004.appx"
    Write-Host "OK"

    if (!$dry_run) {
        Add-AppxPackage -Path "$tmp\ubuntu2004.appx"
    }
}

function Install-CudaToolkit {
    param (
        [string] $tmp = "C:\tmp",
        [string] $output = "cuda_11.2.0.exe",
        [bool] $no_restart = $false
    )

    Prepare

    if (!(Test-Path -Path "$tmp\$output")) {
        Write-Host "Downloading CUDA Toolkit..."
        $cuda_toolkit_uri = "https://developer.download.nvidia.com/compute/cuda/11.2.0/network_installers/cuda_11.2.0_win10_network.exe"
        explain -m "Now we will download CUDA Toolkit from $cuda_toolkit_uri"
        if (!$dry_run) {
            Invoke-WebRequest $cuda_toolkit_uri -OutFile "$tmp\$output"
        }
        Write-Host "OK"
    }

    Write-Host "Installing CUDA Toolkit..."
    explain -m "Now we will start installer process."
    if (!$dry_run) {
        Start-Process -Wait -FilePath "$tmp\$output" -ArgumentList "/S" -PassThru
    }
    Write-Host "Wait untill installation finishes..."
    waitForPress
    Write-Host "OK"

    Cleanup

    if (!$no_restart) {
        Write-Host "Restarting..."
        if (!$dry_run) {
            Restart-Computer
        }
    }
}

function Install-cuDNN {
    param (
        [string] $tmp = "C:\tmp",
        [string] $target = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2",
        [string] $output = "cudnn.zip"
    )

    Prepare

    if (!(Test-Path -Path "$tmp\$output")) {
        Write-Host "Downloading cuDNN..."
        $cudnn_uri = "https://github.com/PolSl-PBL-7/python-template/releases/download/v1.1.0/cudnn.zip"
        explain -m "Now we will download cuDNN from $cudnn_uri"
        if (!$dry_run) {
            Invoke-WebRequest $cudnn_uri -OutFile "$tmp\$output"
        }
        Write-Host "OK"
    }

    Write-Host "Installing cuDNN..."
    explain -m "Now we will extract cuDNN from $tmp\$output and install it into $target"
    if (!$dry_run) {
        Expand-Archive "$tmp\$output" -Force
        Copy-Item "$tmp\cudnn\cuda\bin" -Destination "$target\bin" -Recurse -Force
        Copy-Item "$tmp\cudnn\cuda\include" -Destination "$target\include" -Recurse -Force
        Copy-Item "$tmp\cudnn\cuda\lib\x64" -Destination "$target\lib\x64" -Recurse -Force
    }
    Write-Host "OK"

    Cleanup
}

function Install-ffmpeg {
    param (
        [string] $tmp = "C:\tmp",
        [string] $output = "ffmpeg.7z",
        [string] $target = "C:\bin\ffmpeg"
    )

    Prepare

    if (!(Test-Path -Path "$tmp\$output")) {
        Write-Host "Downloading ffmpeg..."
        $ffmpeg_uri = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z"
        explain -m "Now we will download ffmpeg from $ffmpeg_uri"
        if (!$dry_run) {
            Invoke-WebRequest $ffmpeg_uri -OutFile "$tmp\$output"
        }
        Write-Host "OK"
    }

    Write-Host "Installing ffmpeg..."
    explain -m "Now we will extract ffmpeg from $tmp\$output and install it into $target"
    if (!$dry_run) {
        7z x -aoa "$tmp\$output" -o"$tmp"
        New-Item $target -ItemType "directory" -Force | Out-Null
        Move-Item "$tmp\ffmpeg-*" -Destination $target -Force
    }
    Write-Host "OK"

    Cleanup
}

function Install-All {
    param (
        [string] $tmp = "C:\tmp"
    )

    explain -m "Now we will perform the installation process for CUDA Toolkit"
    Install-CudaToolkit -tmp $tmp -no_restart $true
    explain -m "Now we will perform the installation process for cuDNN"
    Install-cuDNN -tmp $tmp
    explain -m "Now we will perform the installation process for ffmpeg"
    Install-ffmpeg -tmp $tmp

    if (!$dry_run) {
        Write-Host "Restarting..."
        Restart-Computer
    }
}

function Test {
    param (
        [string] $tmp = "C:\tmp"
    )

    explain -m "This is just a test command"
    Write-Host "Test command output. Param tmp = $tmp"
    exit 456
}

if ($test) {
    Test -tmp $tmp
}
if ($install_all) {
    Install-All -tmp $tmp
}
if ($install_cuda) {
    Install-CudaToolkit -tmp $tmp
}
if ($install_cudnn) {
    Install-cuDNN -tmp $tmp
}
if ($install_ffmpeg) {
    Install-ffmpeg -tmp $tmp
}
if ($install_wsls1) {
    Install-WslStep1 -tmp $tmp
}
if ($install_wsls2) {
    Install-WslStep2 -tmp $tmp
}
if ($install_wsls3) {
    Install-WslStep3 -tmp $tmp
}
