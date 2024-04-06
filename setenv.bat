@echo off
@rem How to use: `setenv.bat`

for /f "usebackq delims=" %%A in (`python -m site --user-site`) do set PYTHON_SITE=%%A

set LIBTORCH_USE_PYTORCH=1
set LD_LIBRARY_PATH=%PYTHON_SITE%/torch/lib:%LD_LIBRARY_PATH%

echo LIBTORCH_USE_PYTORCH=%LIBTORCH_USE_PYTORCH%
echo LD_LIBRARY_PATH=%LD_LIBRARY_PATH%
