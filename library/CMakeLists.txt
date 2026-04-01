##################################################################################
#
# MIT License
#
# Copyright(c) 2024-2026 James Sandham
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
  include/linalg_primitives.h
  include/linalg_math.h
  include/linalg_memory.h
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

# Provide an option for the user to enable CUDA
option(ENABLE_CUDA "Enable CUDA compilation if NVIDIA CUDA Toolkit is available" ON)

set(BUILD_WITH_CUDA FALSE CACHE BOOL "Internal flag: Project is building with CUDA support" FORCE)

# Conditionally enable the CUDA language
if(ENABLE_CUDA)
    # The robust way to check for language support before enabling is via CheckLanguage.
    include(CheckLanguage)
    check_language(CUDA) # This sets CMAKE_CUDA_COMPILER if nvcc is found

    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}.")
        enable_language(CUDA) # Now actually enable it

        set(CMAKE_CUDA_ARCHITECTURES "75;80;89" CACHE STRING "Target CUDA architectures for project" FORCE)

        find_package(CUDAToolkit QUIET)
        if (CUDAToolkit_FOUND)
            message(STATUS "CUDA toolkit found: ${CUDAToolkit_FOUND}. Enabling CUDA support.")
            set(BUILD_WITH_CUDA TRUE CACHE BOOL "Internal flag: Project is building with CUDA support" FORCE)
        endif()
    endif()
endif()

add_library(linalglib)

target_sources(linalglib PRIVATE ${LINALGLIB_HEADERS_PUBLIC})

if(BUILD_WITH_CUDA)

  set_target_properties(linalglib PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # Link necessary CUDA libraries for the CUDA-enabled library
  # CUDA::cudart is often implicitly linked by nvcc but explicit is clearer.
  target_link_libraries(linalglib PRIVATE CUDA::cudart)

  # Add a definition for C++ code to know if CUDA is enabled within the library
  target_compile_definitions(linalglib PRIVATE LINALGLIB_HAS_CUDA)
endif()

add_subdirectory(src)

if (WIN32)
  target_link_libraries(linalglib PUBLIC perfetto ws2_32)
else()
  target_link_libraries(linalglib PUBLIC perfetto)
endif()

# Find OpenMP (CXX only, not CUDA)
find_package(OpenMP COMPONENTS CXX)
if (OPENMP_FOUND)
    target_link_libraries(linalglib PRIVATE OpenMP::OpenMP_CXX)
endif()

# Find MPI
find_package(MPI)

message(STATUS "CMAKE_CXX_COMPILER_ID : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_ID : ${CMAKE_CUDA_COMPILER_ID}")

# Compiler options
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(linalglib PRIVATE
      # Options for C++ files (MSVC)
      $<$<COMPILE_LANGUAGE:CXX>:/W4>             # Enable warning level 4 for MSVC
      $<$<COMPILE_LANGUAGE:CXX>:/wd4201>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4100>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
    )
  if (OPENMP_FOUND)
    target_compile_options(linalglib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-openmp:llvm>)
  endif()
else()
  target_compile_options(linalglib PRIVATE -Wall -Wpedantic)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(linalglib PRIVATE
    # Options for CUDA files (NVCC)
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3> # Pass /W3 to the host compiler (MSVC) for CUDA files
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>       # Enable relocatable device code (important for shared libraries)
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:MinSizeRel>>:-O2>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  )
endif()

target_compile_definitions(linalglib PRIVATE -DTRACING_ENABLED)
#target_compile_definitions(linalglib PRIVATE -DTRACING_ENABLED -DCONVERGENCE_LOGGING)
#target_compile_definitions(linalglib PRIVATE -DCONVERGENCE_LOGGING)

target_include_directories(linalglib
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>)
