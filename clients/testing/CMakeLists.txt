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

include(FetchContent)

# Get googletest
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)

# Get yaml-cpp
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

# If the yaml-cpp target was created by FetchContent, mark its includes as
# system and silence noisy warnings from that external dependency only.
if(TARGET yaml-cpp::yaml-cpp)
  # yaml-cpp::yaml-cpp may be an ALIAS target. Resolve the real target so we
  # can set properties on it.
  get_target_property(_aliased_target yaml-cpp::yaml-cpp ALIASED_TARGET)
  if(NOT _aliased_target STREQUAL "_yamlcpp_includes-NOTFOUND" AND NOT "${_aliased_target}" STREQUAL "")
    set(_yaml_target ${_aliased_target})
  else()
    set(_yaml_target yaml-cpp::yaml-cpp)
  endif()

  # Try to re-declare its include directories as SYSTEM so compilers suppress
  # diagnostics from third-party headers. This duplicates the INTERFACE
  # include directories as SYSTEM-interface entries on the real target.
  get_target_property(_yamlcpp_includes ${_yaml_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _yamlcpp_includes STREQUAL "_yamlcpp_includes-NOTFOUND" AND NOT "${_yamlcpp_includes}" STREQUAL "")
    target_include_directories(${_yaml_target} SYSTEM INTERFACE ${_yamlcpp_includes})
  endif()

  # Per-target compile options: suppress warnings when compiling yaml-cpp.
  target_compile_options(${_yaml_target} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/external:W0>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4244>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4267>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4456>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4458>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4018>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4189>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4389>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-w>
  )
endif()

add_executable(test_main test_main.cpp)

target_sources(test_main
  PRIVATE
    test_functions_dispatch.cpp
    test_functions_classical.cpp
    test_functions_krylov.cpp
    test_functions_amg.cpp
    test_functions_multiply_by_vector.cpp
    test_functions_multiply_by_matrix.cpp
    test_functions_triangular_solve.cpp
    test_functions_incomplete_cholesky_factorization.cpp
    test_functions_incomplete_cholesky_factorization_dense.cpp
    test_functions_incomplete_LU_factorization_dense.cpp
    test_functions_sptrsv.cpp
    test_functions_spgeam.cpp
    test_functions_transpose.cpp
    test_functions_transpose_dense.cpp
    test_functions_ruiz_scaling.cpp
    test_functions_symmetric_ruiz_scaling.cpp
    test_functions_csric0.cpp
    test_functions_csrilu0.cpp
    test_functions_tridiagonal_solver.cpp
    test_functions_exclusive_scan.cpp
)

add_subdirectory(tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(test_main PRIVATE -W4 -wd4201 -wd4100 -wd4996)
endif()

target_include_directories(test_main PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>)

target_link_libraries(test_main PRIVATE linalglib)
target_link_libraries(test_main PRIVATE gtest_main gmock_main)
target_link_libraries(test_main PRIVATE yaml-cpp::yaml-cpp)

if(WIN32)
  # linalglib is compiled with /openmp:llvm, so test_main depends on the LLVM
  # OpenMP runtime (libomp140.x86_64.dll). That DLL ships next to cl.exe but is
  # not on PATH, so the executable fails to launch during gtest test discovery.
  # It is an implicit compiler dependency, so CMake does not report it in
  # TARGET_RUNTIME_DLLS; copy it next to the executable explicitly.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(_msvc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_libomp_dll "${_msvc_bin_dir}/libomp140.x86_64.dll")
    if(EXISTS "${_libomp_dll}")
      add_custom_command(TARGET test_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_libomp_dll}" "$<TARGET_FILE_DIR:test_main>")
    endif()
  endif()

  # Copy any CMake-tracked runtime DLL dependencies (e.g. if linalglib ever
  # becomes a shared library). Guarded so it is a no-op when the list is empty,
  # which would otherwise make "cmake -E copy_if_different" fail.
  add_custom_command(TARGET test_main POST_BUILD
    COMMAND "$<$<BOOL:$<TARGET_RUNTIME_DLLS:test_main>>:${CMAKE_COMMAND};-E;copy_if_different;$<TARGET_RUNTIME_DLLS:test_main>;$<TARGET_FILE_DIR:test_main>>"
    COMMAND_EXPAND_LISTS)
endif()

include(GoogleTest)
gtest_discover_tests(test_main WORKING_DIRECTORY $<TARGET_FILE_DIR:test_main> DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")


# Create a list of all files with the extension ".yaml"
file(GLOB YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.yaml")

foreach( FILE ${YAML_FILES})
  get_filename_component(FILENAME ${FILE} NAME)
  configure_file("${FILE}" "${PROJECT_BINARY_DIR}/clients/testing/tests/${FILENAME}" COPYONLY)
endforeach()

file(COPY ../matrices DESTINATION ${PROJECT_BINARY_DIR}/clients/testing )
