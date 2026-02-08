@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  MPI Debug Launcher
::  Builds the project, starts mpiexec with --wait-for-debugger,
::  reads PID files written by each rank, and patches launch.json
::  so F5 attaches directly.
::
::  Usage:  start_mpi_debug.bat [num_ranks]
::          Default: 4 ranks
:: ============================================================

:: --------------- Configuration ---------------
set "NUM_RANKS=%~1"
if "!NUM_RANKS!"=="" set NUM_RANKS=4
set "EXE=%~dp0..\build\debug\bin\Debug\fvm2d_solver.exe"
set "CONFIG=%~dp0..\example\euler_mesh\control.yaml"
set "LAUNCH_JSON=%~dp0..\.vscode\launch.json"
set "TEMP_JSON=%TEMP%\launch_temp.json"

:: Directory where each rank writes its PID file (rank_0.pid, rank_1.pid, ...)
set "FVM2D_PID_DIR=%TEMP%\fvm2d_pids"

:: --------------- Step 1: Build ---------------
echo.
echo ========================================
echo   Step 1 : CMake Build (Debug)
echo ========================================
C:\Software\CMake\bin\cmake.exe --build --preset debug
if %ERRORLEVEL% neq 0 (
    echo BUILD FAILED
    exit /b 1
)

:: --------------- Step 2: Launch MPI ---------------
echo.
echo ========================================
echo   Step 2 : Start mpiexec  (%NUM_RANKS% ranks)
echo ========================================

:: Clean and create PID directory
if exist "%FVM2D_PID_DIR%" rd /s /q "%FVM2D_PID_DIR%"
mkdir "%FVM2D_PID_DIR%"

:: Launch mpiexec in background — ranks will write PID files directly
start /b mpiexec -n %NUM_RANKS% "%EXE%" "%CONFIG%" --wait-for-debugger

echo Waiting for all ranks to write PID files...
set FOUND=0
set ELAPSED=0
:wait_loop
if !FOUND! geq %NUM_RANKS% goto :parse_pids
if !ELAPSED! geq 60 (
    echo WARNING: Only !FOUND!/%NUM_RANKS% ranks reported within 60 s
    goto :parse_pids
)
timeout /t 1 /nobreak >nul
set /a ELAPSED+=1

:: Count how many rank_*.pid files exist
set FOUND=0
for %%F in ("%FVM2D_PID_DIR%\rank_*.pid") do set /a FOUND+=1
goto :wait_loop

:: --------------- Step 3: Parse PIDs ---------------
:parse_pids
echo.
echo ========================================
echo   Step 3 : Parse PIDs ^& patch launch.json
echo ========================================

:: Read PID from each file
set RANK_COUNT=0
for /L %%R in (0,1,%NUM_RANKS%) do (
    if %%R lss %NUM_RANKS% (
        set "PID_FILE=%FVM2D_PID_DIR%\rank_%%R.pid"
        if exist "!PID_FILE!" (
            set /p PID_%%R=<"!PID_FILE!"
            echo   Rank %%R  PID = !PID_%%R!
            set /a RANK_COUNT+=1
        )
    )
)

:: --------------- Step 4: Patch launch.json ---------------
:: Strategy:
::   1. Find where "Attach Rank" blocks begin (or the manual "Attach to MPI Rank")
::   2. Keep everything before that cut point
::   3. Append fresh Attach Rank entries from PID files
::   4. Close the JSON array/object

:: -- Find cut point (first "Attach" entry) --
set FIRST_ATTACH=0
for /f "tokens=1 delims=:" %%N in ('findstr /n /c:"Attach" "!LAUNCH_JSON!"') do (
    if !FIRST_ATTACH! equ 0 set FIRST_ATTACH=%%N
)

if !FIRST_ATTACH! gtr 0 (
    :: Existing attach entries: cut at the opening { of the first one
    set /a CUTOFF=FIRST_ATTACH - 1
) else (
    :: No attach entries yet: cut at the closing ] of configurations array
    for /f "tokens=1 delims=:" %%N in ('findstr /n /c:"]" "!LAUNCH_JSON!"') do (
        set CUTOFF=%%N
    )
)

:: -- Copy lines 1..(CUTOFF-1) to temp, buffering the last line --
:: We buffer the last line so we can ensure it ends with a comma.
type nul >"!TEMP_JSON!"
set IS_FIRST=1
set "LAST_LINE="
for /f "tokens=1* delims=:" %%A in ('findstr /n "^^" "!LAUNCH_JSON!"') do (
    set /a LN=%%A
    if !LN! lss !CUTOFF! (
        if !IS_FIRST! equ 0 (
            >>"!TEMP_JSON!" echo(!LAST_LINE!
        )
        set "LAST_LINE=%%B"
        set IS_FIRST=0
    )
)

:: Ensure trailing comma before new entries
if !FIRST_ATTACH! equ 0 (
    set "LAST_LINE=!LAST_LINE:}=},!"
)
>>"!TEMP_JSON!" echo(!LAST_LINE!

:: -- Append new Attach Rank entries --
set WRITTEN=0
for /L %%R in (0,1,%NUM_RANKS%) do (
    if %%R lss %NUM_RANKS% (
        set "PID_FILE=%FVM2D_PID_DIR%\rank_%%R.pid"
        if exist "!PID_FILE!" (
            set /a WRITTEN+=1
            >>"!TEMP_JSON!" echo         {
            >>"!TEMP_JSON!" echo             "name": "Attach Rank %%R",
            >>"!TEMP_JSON!" echo             "type": "cppvsdbg",
            >>"!TEMP_JSON!" echo             "request": "attach",
            >>"!TEMP_JSON!" echo             "processId": "!PID_%%R!"
            if !WRITTEN! lss !RANK_COUNT! (
                >>"!TEMP_JSON!" echo         },
            ) else (
                >>"!TEMP_JSON!" echo         }
            )
        )
    )
)

:: -- Close JSON --
>>"!TEMP_JSON!" echo     ]
>>"!TEMP_JSON!" echo }

:: -- Replace original --
move /y "!TEMP_JSON!" "!LAUNCH_JSON!" >nul

echo.
echo launch.json updated successfully.
echo.
echo ========================================
echo   MPI ranks are waiting for debugger.
echo   Select "Attach Rank X" and press F5.
echo ========================================
echo Press Ctrl+C to stop mpiexec.

:: --------------- Keep alive ---------------
:keep_alive
timeout /t 5 /nobreak >nul
tasklist /fi "imagename eq fvm2d_solver.exe" 2>nul | findstr /i "fvm2d_solver" >nul
if %ERRORLEVEL% equ 0 goto :keep_alive

echo.
echo MPI processes have exited.

:: Clean up PID files
rd /s /q "%FVM2D_PID_DIR%" 2>nul
