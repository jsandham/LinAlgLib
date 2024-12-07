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
# The above copyright noticeand this permission notice shall be included in all
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
#find_package(gtest REQUIRED)

set(LINALGLIB_CLIENTS_TEST_SOURCE
  test_iterative.cpp
  test_pcg.cpp
  test_rsamg.cpp
  test_saamg.cpp
)

set(LINALGLIB_CLIENTS_COMMON_SOURCE
  ../common/utility.cpp
)

add_executable(test_main test_main.cpp ${LINALGLIB_CLIENTS_TEST_SOURCE} ${LINALGLIB_CLIENTS_COMMON_SOURCE})

target_compile_options(test_main PRIVATE /FS /MTd /Zi)

target_include_directories(test_main PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>)
target_include_directories(test_main PRIVATE $<BUILD_INTERFACE:C:\\Users\\James\\Documents\\googletest\\googletest\\include>)

target_link_libraries(test_main PRIVATE linalglib)
target_link_libraries(test_main PRIVATE "C:\\Users\\James\\Documents\\googletest\\build\\lib\\Debug\\gtest.lib")

file(COPY ../matrices DESTINATION ${PROJECT_BINARY_DIR}/tests )