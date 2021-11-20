.\scripts\install.ps1 -ffmpeg

py -3.9 -m venv venv

.\venv\Scripts\Activate.ps1

py -3.9 .\scripts\bootstrap.py

py -3.9 -m pip install -r .\requirements.txt
