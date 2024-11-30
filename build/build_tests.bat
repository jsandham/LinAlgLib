@echo off

setlocal enabledelayedexpansion

set WARN=-W4 -wd4201 -wd4100 -wd4996
set OPT=/Od
set MODEFLAGS=/FS /MTd /Zi
set MODE=debug
set FLAGS=-nologo /EHsc

set GTEST_INCLUDE="C:\Users\James\Documents\googletest\googletest\include"
set GTEST_LIB="C:\Users\James\Documents\googletest\build\lib"

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
		set MODEFLAGS=/FS /MTd /Zi
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
call cl /std:c++17 %OPT% %WARN% %MODEFLAGS% %FLAGS% /I"../library/include" /I"../clients/common" /I%GTEST_INCLUDE% /I"../clients/tests" "../clients/tests/test_main.cpp" "lib/debug/linalg.lib" "../clients/common/utility.cpp" %GTEST_LIB%"/Debug/gtest.lib" "../clients/tests/test_iterative.cpp" "../clients/tests/test_pcg.cpp" "../clients/tests/test_saamg.cpp" "../clients/tests/test_rsamg.cpp"