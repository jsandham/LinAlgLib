##################################################################################
#
# MIT License
#
# Copyright(c) 2025 James Sandham
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
FetchContent_Declare(perfetto
        GIT_REPOSITORY https://github.com/google/perfetto.git
        GIT_TAG v50.1)
FetchContent_MakeAvailable(perfetto)

find_package(Threads)

# Define a static library for Perfetto.
include_directories(${CMAKE_BINARY_DIR}/_deps/perfetto-src/sdk)
add_library(perfetto STATIC ${CMAKE_BINARY_DIR}/_deps/perfetto-src/sdk/perfetto.cc)

target_include_directories(perfetto PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/perfetto-src/sdk>)

if (WIN32)
  # The perfetto library contains many symbols, so it needs the big object format.
  target_compile_options(perfetto PRIVATE "/bigobj")
  # Disable legacy features in windows.h.
  target_compile_definitions(perfetto PRIVATE -DWIN32_LEAN_AND_MEAN -DNOMINMAX)
endif (WIN32)
