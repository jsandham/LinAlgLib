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
  include/slaf.h
  include/EigenValueSolvers/power_iteration.h
  include/IterativeSolvers/AMG/amg_aggregation.h
  include/IterativeSolvers/AMG/amg_strength.h
  include/IterativeSolvers/AMG/amg_util.h
  include/IterativeSolvers/AMG/amg.h
  include/IterativeSolvers/AMG/rsamg_old.h
  include/IterativeSolvers/AMG/rsamg.h
  include/IterativeSolvers/AMG/saamg.h
  include/IterativeSolvers/AMG/uaamg.h
  include/IterativeSolvers/Classic/gauss_seidel.h
  include/IterativeSolvers/Classic/jacobi.h
  include/IterativeSolvers/Classic/richardson.h
  include/IterativeSolvers/Classic/sor.h
  include/IterativeSolvers/Classic/ssor.h
  include/IterativeSolvers/Classic/symmetric_gauss_seidel.h
  include/IterativeSolvers/Krylov/bicgstab.h
  include/IterativeSolvers/Krylov/gmres.h
  include/IterativeSolvers/Krylov/cg.h
  include/IterativeSolvers/Preconditioner/preconditioner.h
  include/vector.h
  include/csr_matrix.h
  include/linalg.h
)

set(LINALGLIB_SOURCE
  src/slaf.cpp
  src/EigenValueSolvers/power_iteration.cpp
  src/IterativeSolvers/AMG/amg_aggregation.cpp
  src/IterativeSolvers/AMG/amg_strength.cpp
  src/IterativeSolvers/AMG/amg_util.cpp
  src/IterativeSolvers/AMG/amg.cpp
  src/IterativeSolvers/AMG/rsamg_old.cpp
  src/IterativeSolvers/AMG/rsamg.cpp
  src/IterativeSolvers/AMG/saamg.cpp
  src/IterativeSolvers/AMG/uaamg.cpp
  src/IterativeSolvers/Classic/gauss_seidel.cpp
  src/IterativeSolvers/Classic/jacobi.cpp
  src/IterativeSolvers/Classic/richardson.cpp
  src/IterativeSolvers/Classic/sor.cpp
  src/IterativeSolvers/Classic/ssor.cpp
  src/IterativeSolvers/Classic/symmetric_gauss_seidel.cpp
  src/IterativeSolvers/Krylov/bicgstab.cpp
  src/IterativeSolvers/Krylov/gmres.cpp
  src/IterativeSolvers/Krylov/cg.cpp
  src/IterativeSolvers/Preconditioner/preconditioner.cpp
  src/IterativeSolvers/iter_control.cpp
  src/vector.cpp
  src/csr_matrix.cpp
  src/perfetto_trace.cpp
)

add_library(linalglib ${LINALGLIB_SOURCE} ${LINALGLIB_HEADERS_PUBLIC})

if (WIN32)
  target_link_libraries(linalglib PUBLIC perfetto ws2_32)
else()
  target_link_libraries(linalglib PUBLIC perfetto)
endif()

# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(linalglib PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find MPI
find_package(MPI)

# Compiler options
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if (OPENMP_FOUND)
    target_compile_options(linalglib PRIVATE -openmp:llvm -W4 -wd4201 -wd4100 -wd4996)
  else()
    target_compile_options(linalglib PRIVATE -W4 -wd4201 -wd4100 -wd4996)
  endif()
else()
  target_compile_options(linalglib PRIVATE -Wall -Wpedantic)
endif()

# target_compile_definitions(linalglib PRIVATE -DTRACING_ENABLED -DCONVERGENCE_LOGGING)
target_compile_definitions(linalglib PRIVATE -DCONVERGENCE_LOGGING)

target_include_directories(linalglib
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>)

