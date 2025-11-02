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

set(LINALGLIB_CLIENTS_TEST_SOURCE
  test_functions_dispatch.cpp

  test_functions_classical.cpp
  test_functions_krylov.cpp
  test_functions_amg.cpp
  test_functions_spmv.cpp
  test_functions_spgemm.cpp
  test_functions_spgeam.cpp
  test_functions_transpose.cpp
  test_functions_exclusive_scan.cpp

  tests/test_jacobi.cpp
  tests/test_gauss_seidel.cpp
  tests/test_symmetric_gauss_seidel.cpp
  tests/test_SOR.cpp
  tests/test_SSOR.cpp

  tests/test_CG.cpp
  tests/test_GMRES.cpp
  tests/test_BICGSTAB.cpp

  tests/test_UAAMG.cpp
  tests/test_SAAMG.cpp
  #tests/test_RSAMG.cpp

  tests/test_SpMV.cpp
  tests/test_SpGEMM.cpp
  tests/test_transpose.cpp
  tests/test_exclusive_scan.cpp
)

set(LINALGLIB_CLIENTS_COMMON_SOURCE
  ../common/utility.cpp
)

add_executable(test_main test_main.cpp ${LINALGLIB_CLIENTS_TEST_SOURCE} ${LINALGLIB_CLIENTS_COMMON_SOURCE})

if(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
  target_compile_options(test_main PRIVATE -W4 -wd4201 -wd4100 -wd4996)
endif()

target_include_directories(test_main PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>)

target_link_libraries(test_main PRIVATE linalglib)
target_link_libraries(test_main PRIVATE gtest_main gmock_main)
target_link_libraries(test_main PRIVATE yaml-cpp::yaml-cpp)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")


# Create a list of all files with the extension ".yaml"
file(GLOB YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.yaml")

foreach( FILE ${YAML_FILES})
  get_filename_component(FILENAME ${FILE} NAME)
  configure_file("${FILE}" "${PROJECT_BINARY_DIR}/clients/testing/tests/${FILENAME}" COPYONLY)
endforeach()

file(COPY ../matrices DESTINATION ${PROJECT_BINARY_DIR}/clients/testing )
