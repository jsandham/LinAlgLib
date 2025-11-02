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
add_library(perfetto STATIC ${CMAKE_BINARY_DIR}/_deps/perfetto-src/sdk/perfetto.cc)

# Mark perfetto's headers as SYSTEM to suppress warnings coming from
# third-party headers. We still expose the include path to consumers via
# PUBLIC so other targets that link against perfetto can find its headers.
target_include_directories(perfetto SYSTEM PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/_deps/perfetto-src/sdk>)

if (WIN32)
  # The perfetto library contains many symbols, so it needs the big object format.
  target_compile_options(perfetto PRIVATE "/bigobj")
  # Disable legacy features in windows.h.
  target_compile_definitions(perfetto PRIVATE -DWIN32_LEAN_AND_MEAN -DNOMINMAX)
endif (WIN32)

# Silence compiler warnings originating from the perfetto dependency only.
# We do this per-target so warnings from the main project are still shown.
# For MSVC use /W0, for GCC/Clang use -w.
target_compile_options(perfetto PRIVATE
  # For MSVC, prefer suppressing warnings coming from external/system headers
  # using /external:W0; this works better with SYSTEM includes than /W0.
  $<$<CXX_COMPILER_ID:MSVC>:/external:W0>
  # Additionally disable a small set of noisy warning IDs that originate
  # from third-party or STL instantiations when compiling perfetto.
  $<$<CXX_COMPILER_ID:MSVC>:/wd4244>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4267>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4456>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4458>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4018>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4189>
  $<$<CXX_COMPILER_ID:MSVC>:/wd4389>
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-w>
)
