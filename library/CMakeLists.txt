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
  include/linalg_math.h
  include/eigen_value_solvers/power_iteration.h
  include/iterative_solvers/amg/amg_aggregation.h
  include/iterative_solvers/amg/amg_strength.h
  include/iterative_solvers/amg/amg_util.h
  include/iterative_solvers/amg/amg.h
  include/iterative_solvers/amg/rsamg_old.h
  include/iterative_solvers/amg/rsamg.h
  include/iterative_solvers/amg/saamg.h
  include/iterative_solvers/amg/uaamg.h
  include/iterative_solvers/classic/gauss_seidel.h
  include/iterative_solvers/classic/jacobi.h
  include/iterative_solvers/classic/richardson.h
  include/iterative_solvers/classic/sor.h
  include/iterative_solvers/classic/ssor.h
  include/iterative_solvers/classic/symmetric_gauss_seidel.h
  include/iterative_solvers/krylov/bicgstab.h
  include/iterative_solvers/krylov/gmres.h
  include/iterative_solvers/krylov/cg.h
  include/iterative_solvers/preconditioner/preconditioner.h
  include/vector.h
  include/csr_matrix.h
  include/linalg.h
)

set(LINALGLIB_SOURCE
  src/backend/host/math.cpp
  src/backend/device/math.cpp
  src/linalg_math.cpp
  src/eigen_value_solvers/power_iteration.cpp
  src/iterative_solvers/amg/amg_aggregation.cpp
  src/iterative_solvers/amg/amg_strength.cpp
  src/iterative_solvers/amg/amg_util.cpp
  src/iterative_solvers/amg/amg.cpp
  src/iterative_solvers/amg/rsamg_old.cpp
  src/iterative_solvers/amg/rsamg.cpp
  src/iterative_solvers/amg/saamg.cpp
  src/iterative_solvers/amg/uaamg.cpp
  src/iterative_solvers/classic/gauss_seidel.cpp
  src/iterative_solvers/classic/jacobi.cpp
  src/iterative_solvers/classic/richardson.cpp
  src/iterative_solvers/classic/sor.cpp
  src/iterative_solvers/classic/ssor.cpp
  src/iterative_solvers/classic/symmetric_gauss_seidel.cpp
  src/iterative_solvers/krylov/bicgstab.cpp
  src/iterative_solvers/krylov/gmres.cpp
  src/iterative_solvers/krylov/cg.cpp
  src/iterative_solvers/preconditioner/preconditioner.cpp
  src/iterative_solvers/iter_control.cpp
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

