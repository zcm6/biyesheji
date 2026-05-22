$ErrorActionPreference = "Stop"

pyinstaller "通信系统仿真教学演示.spec" --clean -y

Write-Host ""
Write-Host "Build complete:"
Write-Host "dist\通信系统仿真教学演示\通信系统仿真教学演示.exe"
