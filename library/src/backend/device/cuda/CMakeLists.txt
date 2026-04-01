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
    cuda_amg_aggregation.cu
    cuda_amg_strength.cu
    cuda_axpy.cu
    cuda_matrix_vector.cu
    cuda_ssor.cu
    cuda_extract.cu
    cuda_scale.cu
    cuda_csric0.cu
    cuda_csrilu0.cu
    cuda_csr2csc.cu
    cuda_csrtrsv.cu
    cuda_csrgemm.cu
    cuda_csrgeam.cu
    cuda_tridiagonal.cu
    cuda_math.cu
    cuda_memory.cu
    cuda_primitives.cu
)
