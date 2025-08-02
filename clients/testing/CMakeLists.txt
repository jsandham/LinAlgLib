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

set(LINALGLIB_CLIENTS_TEST_SOURCE
  test_functions_classical.cpp
  test_functions_krylov.cpp
  test_functions_amg.cpp
  test_functions_dispatch.cpp

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
