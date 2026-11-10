@echo off
rem ---------------------------------------------------------------------------
rem Device build: CUDA 12.6 (nvcc) + Visual Studio 2022 BuildTools (MSVC 14.44).
rem
rem Two toolchains on one machine and no PATH surgery: vcvars64 from the BuildTools
rem installation is called here, and CMAKE_CUDA_COMPILER is pinned to the 12.6
rem toolkit so the build cannot silently pick up the 12.1 that is first on PATH.
rem
rem NVCC_PREPEND_FLAGS=--allow-unsupported-compiler: CUDA 12.6 does not know
rem MSVC 14.44 and refuses to compile without it. The flag is safe here because the
rem host code is also compiled by /W4 /WX through -Xcompiler (see CMakeLists.txt).
rem
rem Usage:  build-device.bat [clean]
rem ---------------------------------------------------------------------------
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "NVCC_PREPEND_FLAGS=--allow-unsupported-compiler"
set "BUILD=%~dp0build-dev"
if /i "%~1"=="clean" rmdir /s /q "%BUILD%" 2>nul

cmake -S "%~dp0." -B "%BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DPOOLKIT_BUILD_TESTS=ON ^
    -DPOOLKIT_BUILD_TOOLS=ON ^
    -DPOOLKIT_WITH_CUDA=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD%"
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD%" --output-on-failure --timeout 900
exit /b %errorlevel%
