@echo off

setlocal enabledelayedexpansion

set WARN=-W4 -wd4201 -wd4100 -wd4996
set OPT=/Od
set MODEFLAGS=/FS /MDd /Zi
set MODE=debug
set FLAGS=-nologo /EHsc

:: run through batch file parameter inputs
for %%x in (%*) do (
	if "%%x"=="/help" (
		echo "help"
	)
	if "%%x"=="/O2" (
		set OPT=/O2
	)
	if "%%x"=="/debug" (
		set MODE=debug
		set MODEFLAGS=/FS /MDd /Zi
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/FS /MD 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
echo [92mOptimization level: %OPT%[0m	

:: compile c++ examples
for /R "../clients/examples/" %%f in (*.cpp) do (
	echo [92mCompiling C++ example...%%f[0m
	call cl /std:c++17 %OPT% %WARN% %MODEFLAGS% %FLAGS% /I"../library/include" /I"../clients/common" %%f "lib/debug/linalg.lib" "../clients/common/utility.cpp"
)