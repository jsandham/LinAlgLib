##################################################################################
#
# MIT License
#
# Copyright(c) 2024 James Sandham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this softwareand associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
##################################################################################

set(LINALGLIB_HEADERS_PUBLIC
  include/LinearSolvers/slaf.h
  include/EigenValueSolvers/power_iteration.h
  include/LinearSolvers/AMG/amg_aggregation.h
  include/LinearSolvers/AMG/amg_strength.h
  include/LinearSolvers/AMG/amg.h
  include/LinearSolvers/AMG/rsamg_old.h
  include/LinearSolvers/AMG/rsamg.h
  include/LinearSolvers/AMG/saamg.h
  include/LinearSolvers/AMG/uaamg.h
  include/LinearSolvers/Classic/gauss_seidel.h
  include/LinearSolvers/Classic/jacobi.h
  include/LinearSolvers/Classic/richardson.h
  include/LinearSolvers/Classic/sor.h
  include/LinearSolvers/Classic/ssor.h
  include/LinearSolvers/Classic/symmetric_gauss_seidel.h
  include/LinearSolvers/Krylov/pbicgstab.h
  include/LinearSolvers/Krylov/bicgstab.h
  include/LinearSolvers/Krylov/gmres.h
  include/LinearSolvers/Krylov/pcg.h
  include/LinearSolvers/Krylov/cg.h
  include/LinearSolvers/Preconditioner/preconditioner.h
  include/linalg.h
)

set(LINALGLIB_SOURCE
  src/LinearSolvers/slaf.cpp
  src/EigenValueSolvers/power_iteration.cpp
  src/LinearSolvers/AMG/amg_aggregation.cpp
  src/LinearSolvers/AMG/amg_strength.cpp
  src/LinearSolvers/AMG/amg.cpp
  src/LinearSolvers/AMG/rsamg_old.cpp
  src/LinearSolvers/AMG/rsamg.cpp
  src/LinearSolvers/AMG/saamg.cpp
  src/LinearSolvers/AMG/uaamg.cpp
  src/LinearSolvers/Classic/gauss_seidel.cpp
  src/LinearSolvers/Classic/jacobi.cpp
  src/LinearSolvers/Classic/richardson.cpp
  src/LinearSolvers/Classic/sor.cpp
  src/LinearSolvers/Classic/ssor.cpp
  src/LinearSolvers/Classic/symmetric_gauss_seidel.cpp
  src/LinearSolvers/Krylov/pbicgstab.cpp
  src/LinearSolvers/Krylov/bicgstab.cpp
  src/LinearSolvers/Krylov/gmres.cpp
  src/LinearSolvers/Krylov/pcg.cpp
  src/LinearSolvers/Krylov/cg.cpp
  src/LinearSolvers/Preconditioner/preconditioner.cpp
)

add_library(linalglib ${LINALGLIB_SOURCE} ${LINALGLIB_HEADERS_PUBLIC})

# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(linalglib PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find MPI
find_package(MPI)

# Compiler options
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(linalglib PRIVATE -W4 -wd4201 -wd4100 -wd4996)
else()
  target_compile_options(linalglib PRIVATE -Wall -Wpedantic)
endif()

target_include_directories(linalglib
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>)

