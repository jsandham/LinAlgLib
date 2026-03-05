//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************

#ifndef TRIDIAGONAL_SOLVER_LARGE_KERNELS_H
#define TRIDIAGONAL_SOLVER_LARGE_KERNELS_H

#include <cuda/atomic>

#include "common.cuh"

template <uint32_t BLOCKSIZE, typename T>
__global__ void pcr_tiled_forward_kernel(int m,
                                         int n,
                                         const T* __restrict__ lower,
                                         const T* __restrict__ main,
                                         const T* __restrict__ upper,
                                         const T* __restrict__ B,
                                         T* __restrict__ lower_modified,
                                         T* __restrict__ main_modified,
                                         T* __restrict__ upper_modified,
                                         T* __restrict__ B_modified,
                                         T* __restrict__ spike_lower,
                                         T* __restrict__ spike_main,
                                         T* __restrict__ spike_upper,
                                         T* __restrict__ spike_B)
{
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCKSIZE + tid;

    // Shared memory for the tile's coefficients
    __shared__ T sa[BLOCKSIZE];
    __shared__ T sb[BLOCKSIZE];
    __shared__ T sc[BLOCKSIZE];
    __shared__ T sd[BLOCKSIZE];

    // 1. Load data from Global to Shared Memory
    sa[tid] = (gid < m) ? lower[gid] : static_cast<T>(0);
    sb[tid] = (gid < m) ? main[gid] : static_cast<T>(1);
    sc[tid] = (gid < m) ? upper[gid] : static_cast<T>(0);
    sd[tid] = (gid < m) ? B[gid] : static_cast<T>(0);
    __syncthreads();

    // 2. Perform Local PCR iterations (log2(BLOCKSIZE))
    for(int k = 1; k < BLOCKSIZE; k <<= 1)
    {
        const int left  = tid - k;
        const int right = tid + k;

        const T a_left = (left >= 0) ? sa[left] : static_cast<T>(0);
        const T b_left = (left >= 0) ? sb[left] : static_cast<T>(1);
        const T c_left = (left >= 0) ? sc[left] : static_cast<T>(0);
        const T d_left = (left >= 0) ? sd[left] : static_cast<T>(0);

        const T a_right = (right < BLOCKSIZE) ? sa[right] : static_cast<T>(0);
        const T b_right = (right < BLOCKSIZE) ? sb[right] : static_cast<T>(1);
        const T c_right = (right < BLOCKSIZE) ? sc[right] : static_cast<T>(0);
        const T d_right = (right < BLOCKSIZE) ? sd[right] : static_cast<T>(0);

        const T a = sa[tid];
        const T b = sb[tid];
        const T c = sc[tid];
        const T d = sd[tid];

        // Elimination math
        const T alpha = (left >= 0) ? -a / b_left : static_cast<T>(0);
        const T gamma = (right < BLOCKSIZE) ? -c / b_right : static_cast<T>(0);

        // If neighbors were out of tile, the alpha/gamma remains,
        // effectively preserving the dependency for the Global Glue phase.

        __syncthreads(); // Ensure all reads are done before writing

        sa[tid] = alpha * a_left;
        sb[tid] = b + alpha * c_left + gamma * a_right;
        sc[tid] = gamma * c_right;
        sd[tid] = d + alpha * d_left + gamma * d_right;

        __syncthreads(); // Sync for next iteration
    }

    // --- NEW: Write modified coefficients back to Global Memory ---
    // These modified values represent the relationship:
    // m[i]*x[i] = d[i] - l[i]*x[tile_start] - u[i]*x[tile_end]
    if(gid < m)
    {
        lower_modified[gid] = sa[tid];
        main_modified[gid]  = sb[tid];
        upper_modified[gid] = sc[tid];
        B_modified[gid]     = sd[tid];
    }

    // 3. Write Interface Rows to the Global Glue System
    // Each tile contributes its first and last rows
    if(tid == 0 || tid == BLOCKSIZE - 1)
    {
        // glue_idx: Tile 0 gives index 0,1; Tile 1 gives index 2,3...
        int spike_idx = blockIdx.x * 2 + (tid == 0 ? 0 : 1);

        spike_lower[spike_idx] = sa[tid];
        spike_main[spike_idx]  = sb[tid];
        spike_upper[spike_idx] = sc[tid];
        spike_B[spike_idx]     = sd[tid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void spike_solver_pcr_kernel(int num_spikes, // e.g., 512
                                        const T* __restrict__ l_spike,
                                        const T* __restrict__ m_spike,
                                        const T* __restrict__ u_spike,
                                        const T* __restrict__ B_spike,
                                        T* __restrict__ X_spike_out)
{
    const int tid = threadIdx.x;

    __shared__ T sa[BLOCKSIZE];
    __shared__ T sb[BLOCKSIZE];
    __shared__ T sc[BLOCKSIZE];
    __shared__ T sd[BLOCKSIZE];

    // 1. Load the spike system into shared memory
    sa[tid] = (tid < num_spikes) ? l_spike[tid] : static_cast<T>(0);
    sb[tid] = (tid < num_spikes) ? m_spike[tid] : static_cast<T>(1);
    sc[tid] = (tid < num_spikes) ? u_spike[tid] : static_cast<T>(0);
    sd[tid] = (tid < num_spikes) ? B_spike[tid] : static_cast<T>(0);
    __syncthreads();

    // 2. PCR Algorithm
    // For 512 elements, this loop runs 9 times (2^9 = 512)
    for(int h = 1; h < BLOCKSIZE; h *= 2)
    {
        const int left  = tid - h;
        const int right = tid + h;

        const T a_left = (left >= 0) ? sa[left] : static_cast<T>(0);
        const T b_left = (left >= 0) ? sb[left] : static_cast<T>(1);
        const T c_left = (left >= 0) ? sc[left] : static_cast<T>(0);
        const T d_left = (left >= 0) ? sd[left] : static_cast<T>(0);

        const T a_right = (right < BLOCKSIZE) ? sa[right] : static_cast<T>(0);
        const T b_right = (right < BLOCKSIZE) ? sb[right] : static_cast<T>(1);
        const T c_right = (right < BLOCKSIZE) ? sc[right] : static_cast<T>(0);
        const T d_right = (right < BLOCKSIZE) ? sd[right] : static_cast<T>(0);

        const T a = sa[tid];
        const T b = sb[tid];
        const T c = sc[tid];
        const T d = sd[tid];

        const T k1 = (left >= 0) ? a / b_left : static_cast<T>(0);
        const T k2 = (right < BLOCKSIZE) ? c / b_right : static_cast<T>(0);

        __syncthreads(); // Wait for all threads to finish reading 'old' values

        // Update coefficients
        // If k1/k2 are 0 (out of bounds), the original values are preserved
        sb[tid] = b - k1 * c_left - k2 * a_right;
        sd[tid] = d - k1 * d_left - k2 * d_right;
        sa[tid] = -k1 * a_left;
        sc[tid] = -k2 * c_right;

        __syncthreads(); // Wait for all threads to write 'new' values
    }

    // 3. Final Solution
    // After log2(N) steps, the system is diagonalized: b_i * x_i = d_i
    if(tid < num_spikes)
    {
        X_spike_out[tid] = sd[tid] / sb[tid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void backward_sweep_kernel(int m, T* low, T* main, T* up, T* d, T* spike_res, T* X)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCKSIZE + tid;
    if(gid >= m)
        return;

    // Each internal node i needs its neighbors (i-stride) and (i+stride)
    // In a tiled approach, the "neighbors" for the internal nodes
    // were reduced to the spike values during the forward sweep.

    float sol_top    = spike_res[blockIdx.x * 2];
    float sol_bottom = spike_res[blockIdx.x * 2 + 1];

    // Simple back-substitution using the local modified equations
    // Note: In a production PCR, you'd store the intermediate
    // forward sweep values to reconstruct x[gid] here.

    // Example logic for intermediate rows:
    // x[gid] = (d[gid] - l[gid]*sol_top - u[gid]*sol_bottom) / m[gid];
    X[gid] = (d[gid] - low[gid] * sol_top - up[gid] * sol_bottom) / main[gid];
}

// // Cyclic reduction algorithm using shared memory
// // reduce system down to a 512x512 system, solve that system with
// // PCR in a single block, then back substitute to get the final solution
// // That means we launch the first kernel with 512 blocks, and the second kernel with 1 block
// // So if m = 65536, we launch 512 blocks with each block having blocksize = 128, and then 1
// // block with blocksize = 2 * 512 = 1024 (because each block from the first stage produces 2
// // unknowns for the second stage)
// template <uint32_t BLOCKSIZE, typename T>
// __global__ void cr_forward_sweep_kernel(int n,
//                                         const T* __restrict__ lower,
//                                         const T* __restrict__ main,
//                                         const T* __restrict__ upper,
//                                         const T* __restrict__ B,
//                                         T* __restrict__ lower_pyramid,
//                                         T* __restrict__ main_pyramid,
//                                         T* __restrict__ upper_pyramid,
//                                         T* __restrict__ rhs_pyramid,
//                                         T* __restrict__ lower_spike,
//                                         T* __restrict__ main_spike,
//                                         T* __restrict__ upper_spike,
//                                         T* __restrict__ X_spike)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int gid = bid * BLOCKSIZE + tid;

//     // Shared memory for current working set
//     __shared__ T sa[BLOCKSIZE];
//     __shared__ T sb[BLOCKSIZE];
//     __shared__ T sc[BLOCKSIZE];
//     __shared__ T srhs[BLOCKSIZE];

//     // 1. Load data from Global to Shared Memory
//     if(gid < n)
//     {
//         sa[tid]   = lower[gid];
//         sb[tid]   = main[gid];
//         sc[tid]   = upper[gid];
//         srhs[tid] = B[gid];
//     }
//     else
//     {
//         // Handle cases where n is not a multiple of BLOCKSIZE
//         sa[tid]   = 0;
//         sb[tid]   = 1;
//         sc[tid]   = 0;
//         srhs[tid] = 0;
//     }
//     __syncthreads();

//     // 2. CR Forward Sweep
//     int stride = 1;
//     int level  = 0;

//     // Reduce until only indices 0 and BLOCKSIZE-1 are active
//     for(int len = BLOCKSIZE / 2; len > 1; len /= 2)
//     {
//         if(tid < len)
//         {
//             // 'elim' is the node being removed from the system at this level
//             int elim  = (tid * stride * 2) + stride;
//             int left  = elim - stride;
//             int right = elim + stride;

//             // Store the state of the node to be eliminated in the pyramid
//             // We need these exact values to solve for x[elim] in the backward sweep
//             int pyramid_idx            = (level * n) + (bid * BLOCKSIZE) + elim;
//             lower_pyramid[pyramid_idx] = sa[elim];
//             main_pyramid[pyramid_idx]  = sb[elim];
//             upper_pyramid[pyramid_idx] = sc[elim];
//             rhs_pyramid[pyramid_idx]   = srhs[elim];

//             // Elimination Math: Use 'elim' row to modify 'left' and 'right'
//             T k1 = sc[left] / sb[elim];
//             T k2 = sa[right] / sb[elim];

//             sb[left] -= k1 * sa[elim];
//             srhs[left] -= k1 * srhs[elim];
//             sc[left] = -k1 * sc[elim]; // Link 'left' to the node 'right' of elim

//             sb[right] -= k2 * sc[elim];
//             srhs[right] -= k2 * srhs[elim];
//             sa[right] = -k2 * sa[elim]; // Link 'right' to the node 'left' of elim
//         }
//         stride *= 2;
//         level++;
//         __syncthreads();
//     }

//     // 3. Final Reduction: Couple Row 0 and Row (BLOCKSIZE-1)
//     // After the loop, the block is reduced to just two equations.
//     // We must link them so the Spike system sees a 2x2 block.
//     if(tid == 0)
//     {
//         int first = 0;
//         int last  = BLOCKSIZE - 1;

//         // Take a snapshot of original values to avoid using
//         // partially updated results in the second half of the math
//         T f_a = sa[first], f_b = sb[first], f_c = sc[first], f_d = srhs[first];
//         T l_a = sa[last], l_b = sb[last], l_c = sc[last], l_d = srhs[last];

//         // Row 0 eliminates Row 'last'
//         T k_f       = f_c / l_b;
//         sb[first]   = f_b - k_f * l_a;
//         sc[first]   = -k_f * l_c;
//         srhs[first] = f_d - k_f * l_d;

//         // Row 'last' eliminates Row 'first'
//         T k_l      = l_a / f_b;
//         sb[last]   = l_b - k_l * f_c;
//         sa[last]   = -k_l * f_a;
//         srhs[last] = l_d - k_l * f_d;
//     }
//     __syncthreads();

//     // 4. Output to Spike Arrays
//     if(tid == 0)
//     {
//         int out_idx          = 2 * bid;
//         lower_spike[out_idx] = sa[0]; // Original lower[bid*BLOCKSIZE]
//         main_spike[out_idx]  = sb[0];
//         upper_spike[out_idx] = sc[0]; // Points to next block
//         X_spike[out_idx]     = srhs[0];
//     }
//     else if(tid == BLOCKSIZE - 1)
//     {
//         int out_idx          = 2 * bid + 1;
//         lower_spike[out_idx] = sa[BLOCKSIZE - 1]; // Points to prev block
//         main_spike[out_idx]  = sb[BLOCKSIZE - 1];
//         upper_spike[out_idx] = sc[BLOCKSIZE - 1]; // Original upper[...]
//         X_spike[out_idx]     = srhs[BLOCKSIZE - 1];
//     }
// }

// template <typename T>
// __global__ void spike_solver_pcr_kernel(int num_spikes, // e.g., 512
//                                         const T* __restrict__ l_spike,
//                                         const T* __restrict__ m_spike,
//                                         const T* __restrict__ u_spike,
//                                         const T* __restrict__ d_spike,
//                                         T* __restrict__ x_spike_out)
// {
//     const int tid = threadIdx.x;

//     // Shared memory for the spike system (Size = num_spikes)
//     // For 512 spikes, this is only ~8KB for doubles
//     extern __shared__ char shared_mem[];
//     T*                     sa = (T*)shared_mem;
//     T*                     sb = sa + num_spikes;
//     T*                     sc = sb + num_spikes;
//     T*                     sd = sc + num_spikes;

//     // 1. Load the spike system into shared memory
//     if(tid < num_spikes)
//     {
//         sa[tid] = l_spike[tid];
//         sb[tid] = m_spike[tid];
//         sc[tid] = u_spike[tid];
//         sd[tid] = d_spike[tid];
//     }
//     __syncthreads();

//     // 2. PCR Algorithm
//     // For 512 elements, this loop runs 9 times (2^9 = 512)
//     for(int h = 1; h < num_spikes; h <<= 1)
//     {
//         T a = sa[tid];
//         T b = sb[tid];
//         T c = sc[tid];
//         T d = sd[tid];

//         int left  = tid - h;
//         int right = tid + h;

//         T k1 = 0, k2 = 0;

//         if(left >= 0)
//         {
//             k1 = a / sb[left];
//         }
//         if(right < num_spikes)
//         {
//             k2 = c / sb[right];
//         }

//         __syncthreads(); // Wait for all threads to finish reading 'old' values

//         // Update coefficients
//         // If k1/k2 are 0 (out of bounds), the original values are preserved
//         sb[tid] = b - (left >= 0 ? k1 * sc[left] : 0) - (right < num_spikes ? k2 * sa[right] : 0);
//         sd[tid] = d - (left >= 0 ? k1 * sd[left] : 0) - (right < num_spikes ? k2 * sd[right] : 0);
//         sa[tid] = (left >= 0 ? -k1 * sa[left] : 0);
//         sc[tid] = (right < num_spikes ? -k2 * sc[right] : 0);

//         __syncthreads(); // Wait for all threads to write 'new' values
//     }

//     // 3. Final Solution
//     // After log2(N) steps, the system is diagonalized: b_i * x_i = d_i
//     if(tid < num_spikes)
//     {
//         x_spike_out[tid] = sd[tid] / sb[tid];
//     }
// }

// template <uint32_t BLOCKSIZE, typename T>
// __global__ void cr_backward_sweep_kernel(int n,
//                                          const T* __restrict__ lower_pyramid,
//                                          const T* __restrict__ main_pyramid,
//                                          const T* __restrict__ upper_pyramid,
//                                          const T* __restrict__ rhs_pyramid,
//                                          const T* __restrict__ x_spike_in,
//                                          T* __restrict__ X_final)
// {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int gid = bid * BLOCKSIZE + tid;

//     // Shared memory to store the solving X values
//     __shared__ T sx[BLOCKSIZE];

//     // 1. Initialize shared memory with the boundary values from Phase 2
//     if(tid == 0)
//     {
//         sx[0] = x_spike_in[2 * bid];
//     }
//     else if(tid == BLOCKSIZE - 1)
//     {
//         sx[BLOCKSIZE - 1] = x_spike_in[2 * bid + 1];
//     }
//     __syncthreads();

//     // 2. Backward Sweep
//     // We start from the widest stride used in the forward sweep and shrink it
//     // If BLOCKSIZE is 256, the last forward len was 2, so we start backwards from there.
//     int stride = BLOCKSIZE / 2;
//     int level  = (int)log2f(BLOCKSIZE) - 2; // Match the level count from forward sweep

//     for(int len = 2; len <= BLOCKSIZE / 2; len <<= 1)
//     {
//         // Only threads representing nodes to be 'filled in' are active
//         if(tid < len)
//         {
//             int elim  = (tid * stride * 2) + stride;
//             int left  = elim - stride;
//             int right = elim + stride;

//             // Load the coefficients that were used to eliminate this node
//             int pyramid_idx = (level * n) + (bid * BLOCKSIZE) + elim;

//             T a = lower_pyramid[pyramid_idx];
//             T b = main_pyramid[pyramid_idx];
//             T c = upper_pyramid[pyramid_idx];
//             T d = rhs_pyramid[pyramid_idx];

//             // Solve for x[elim]:
//             // b*x_elim + a*x_left + c*x_right = d
//             // x_elim = (d - a*x_left - c*x_right) / b
//             sx[elim] = (d - a * sx[left] - c * sx[right]) / b;
//         }
//         stride >>= 1;
//         level--;
//         __syncthreads();
//     }

//     // 3. Final Step: Fill the Level 0 gaps
//     // The loop above fills the tree, but we have one last set of
//     // original 'odd' nodes (stride 1) to solve.
//     if(tid < (BLOCKSIZE / 2))
//     {
//         int elim  = (tid * 2) + 1;
//         int left  = elim - 1;
//         int right = elim + 1;

//         // Level 0 of pyramid
//         int pyramid_idx = (bid * BLOCKSIZE) + elim;

//         T a = lower_pyramid[pyramid_idx];
//         T b = main_pyramid[pyramid_idx];
//         T c = upper_pyramid[pyramid_idx];
//         T d = rhs_pyramid[pyramid_idx];

//         sx[elim] = (d - a * sx[left] - c * sx[right]) / b;
//     }
//     __syncthreads();

//     // 4. Write final results to global memory
//     if(gid < n)
//     {
//         X_final[gid] = sx[tid];
//     }
// }

#endif // TRIDIAGONAL_SOLVER_LARGE_KERNELS_H
