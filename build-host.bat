@echo off
rem ---------------------------------------------------------------------------
rem Host-only build: no CUDA toolkit, no GPU, no nvcc.
rem
rem Toolchain: Visual Studio 18 (Community) + Ninja. This is the configuration the
rem "host-only" CI job runs on GitHub's ubuntu-22.04 / windows-2022 runners, and
rem it is the one that proves the allocator core is testable without a GPU.
rem
rem Usage:  build-host.bat [clean]
rem    clean   delete the build directory first (a from-scratch build + test)
rem ---------------------------------------------------------------------------
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "BUILD=%~dp0build-host"
if /i "%~1"=="clean" rmdir /s /q "%BUILD%" 2>nul

cmake -S "%~dp0." -B "%BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPOOLKIT_BUILD_TESTS=ON ^
    -DPOOLKIT_BUILD_TOOLS=ON ^
    -DPOOLKIT_WITH_CUDA=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD%"
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD%" --output-on-failure --timeout 600
exit /b %errorlevel%
