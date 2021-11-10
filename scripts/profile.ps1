function Prompt
{
    "(session) " + (Get-Location) + [Environment]::NewLine + "> "
}

$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64"
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include"
$env:PATH += ";C:\Program Files\ffmpeg\bin"
$env:PATH += ";C:\tools\cuda\bin"
