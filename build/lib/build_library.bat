@echo off

setlocal enabledelayedexpansion

set WARN=-W4 -wd4201 -wd4100 -wd4996
set OPENMP=
set OPT=/Od
set MODEFLAGS=/FS /MDd /Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
set MODE=debug
set FLAGS=-nologo /EHsc

:: run through batch file parameter inputs
for %%x in (%*) do (
	if "%%x"=="/help" (
		echo "help"
	)
	if "%%x"=="/omp" (
		set OPENMP=/openmp
	)
	if "%%x"=="/O2" (
		set OPT=/O2
	)
	if "%%x"=="/debug" (
		set MODE=debug
		set MODEFLAGS=/FS /MDd /Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/FS /MD /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
if defined OPENMP (
	echo [92mOpenMP: on[0m	
)else (
	echo [92mOpenMP: off[0m	
)
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code
echo [92mCompiling C++ library code...[0m
for /R "../../library/src/" %%f in (*.cpp) do (
	call cl /c /std:c++17 %OPT% %OPENMP% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
::echo [92mCompiled objects...[0m
::set OBJ_FILES=
::for /r "%MODE%/obj" %%v in (*.obj) do (
::	call :concat_obj %%v
::)

:: create static library
echo [92mCreating static library...[0m
::lib /nologo /out:%MODE%/engine.lib %OBJ_FILES%
lib /nologo /out:%MODE%/linalg.lib %MODE%/obj/*.obj

:: delete .obj fles
::echo [92mDeleting objects...[0m
::set OBJ_FILES=
::for /r "%MODE%/obj" %%v in (*.obj) do (
::	del /s %%v
::)

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof