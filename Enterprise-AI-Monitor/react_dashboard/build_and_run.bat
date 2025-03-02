@echo off
setlocal

echo ===== Enterprise AI Monitor - Build and Run API Bridge =====
echo.

:: Create and navigate to build directory
if not exist build mkdir build
cd build

:: Configure with CMake
echo Running CMake configuration...
cmake .. -DCURL_DISABLED=ON

:: Build the project
echo.
echo Building the project...
cmake --build . --config Release

:: Check if the build was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed! See error messages above.
    goto end
)

echo.
echo Build successful!
echo.
echo === Running API Bridge (mock mode) ===
echo.

:: Run the API bridge
.\Release\api_bridge.exe

:end
echo.
echo Press any key to exit...
pause > nul
endlocal