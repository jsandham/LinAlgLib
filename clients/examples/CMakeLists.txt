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

function(add_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  
  set(LINALGLIB_CLIENTS_COMMON_SOURCE
  ../common/utility.cpp
  )
  
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${LINALGLIB_CLIENTS_COMMON_SOURCE})

  # Target compile options
  if(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
    target_compile_options(${EXAMPLE_TARGET} PRIVATE -W4 -wd4201 -wd4100 -wd4996)
  endif()
  
  # Include common client headers
  target_include_directories(${EXAMPLE_TARGET} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>)

  # Linker dependencies
  target_link_libraries(${EXAMPLE_TARGET} PRIVATE linalglib)
endfunction()

add_example(gauss_seidel_example.cpp)
add_example(gmres_example.cpp)
add_example(jacobi_example.cpp)
add_example(pcg_example.cpp)
add_example(richardson_example.cpp)
add_example(rsamg_example.cpp)
add_example(rsamg_old_example.cpp)
add_example(saamg_example.cpp)
add_example(sor_example.cpp)
add_example(ssor_example.cpp)
add_example(symm_gauss_seidel_example.cpp)
add_example(uaamg_example.cpp)

file(COPY ../matrices DESTINATION ${PROJECT_BINARY_DIR}/clients/examples )