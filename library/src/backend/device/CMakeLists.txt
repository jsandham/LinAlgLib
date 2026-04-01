##################################################################################
#
# MIT License
#
# Copyright(c) 2026 James Sandham
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

target_sources(linalglib
  PRIVATE
    device_amg_aggregation.cpp
    device_amg_strength.cpp
    device_primitives.cpp
    device_math.cpp
    device_axpy.cpp
    device_matrix_vector.cpp
    device_extract.cpp
    device_scale.cpp
    device_ssor.cpp
    device_csr2csc.cpp
    device_csrtrsv.cpp
    device_csrgeam.cpp
    device_csrgemm.cpp
    device_csric0.cpp
    device_csrilu0.cpp
    device_tridiagonal.cpp
    device_vector.cpp
    device_memory.cpp
)

if(BUILD_WITH_CUDA)
  add_subdirectory(cuda)
endif()
