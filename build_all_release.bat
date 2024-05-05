@echo off

if not defined DevEnvDir (
	call "shell.bat"
)

echo [94mBuilding library...[0m
cd "%~dp0\build\lib"
call "build_library.bat" /release /O2
cd "..\.."

echo [94mBuilding examples...[0m
cd "%~dp0\build"
call "build_examples.bat" /release /O2
cd ".."
