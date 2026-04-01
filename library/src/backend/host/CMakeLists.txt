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
    host_amg_aggregation.cpp
    host_amg_strength.cpp
    host_primitives.cpp
    host_math.cpp
    host_axpy.cpp
    host_matrix_vector.cpp
    host_extract.cpp
    host_scale.cpp
    host_ssor.cpp
    host_csr2csc.cpp
    host_csrtrsv.cpp
    host_csrgeam.cpp
    host_csrgemm.cpp
    host_csric0.cpp
    host_csrilu0.cpp
    host_tridiagonal.cpp
    host_vector.cpp
    host_memory.cpp
)
