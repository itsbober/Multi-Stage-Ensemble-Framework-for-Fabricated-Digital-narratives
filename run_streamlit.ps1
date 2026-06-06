Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LogPath = Join-Path $ProjectRoot "reports\streamlit_server.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Set-Content -Path $LogPath -Value ""

$Command = ".\.venv\Scripts\python.exe -m streamlit run .\app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false >> reports\streamlit_server.log 2>&1"
& cmd.exe /c $Command
