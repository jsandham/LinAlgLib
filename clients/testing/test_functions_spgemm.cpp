//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

#include "test_functions.h"
#include "utility.h"

#include <chrono>
#include <cmath>
#include <iostream>

#include "linalg.h"

using namespace linalg;

bool Testing::test_spgemm(Arguments arg)
{
    // 2 1 0 1
    // 1 2 1 0
    // 0 1 2 1
    // 1 0 1 2
    int                 m            = 4;
    int                 n            = 4;
    int                 nnz          = 12;
    std::vector<int>    hcsr_row_ptr = {0, 3, 6, 9, 12};
    std::vector<int>    hcsr_col_ind = {0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3};
    std::vector<double> hcsr_val     = {2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0};

    csr_matrix mat_A(hcsr_row_ptr, hcsr_col_ind, hcsr_val, m, n, nnz);
    csr_matrix mat_B(hcsr_row_ptr, hcsr_col_ind, hcsr_val, m, n, nnz);

    csr_matrix mat_C;

    mat_A.move_to_device();
    mat_B.move_to_device();
    mat_C.move_to_device();

    //mat_A.multiply_by_matrix(mat_C, mat_B);
    matrix_matrix_addition(mat_C, mat_A, mat_B);

    mat_A.move_to_host();
    mat_B.move_to_host();
    mat_C.move_to_host();

    mat_A.print_matrix("A");
    mat_B.print_matrix("B");
    mat_C.print_matrix("C");

    mesh mesh;
    load_gmsh_mesh("../../../../../fluidfe/meshes/mesh1.msh", mesh);

    for(size_t i = 0; i < mesh.nodes.size(); i++)
    {
        const node& n = mesh.nodes[i];
        std::cout << "Node " << n.tag << ": (" << n.x << ", " << n.y << ", " << n.z << ")"
                  << std::endl;
    }

    /*csr_matrix mat_A;
    mat_A.read_mtx(arg.filename);

    for(int i = 0; i < mat_A.get_nnz(); i++)
    {
        double* csr_val = mat_A.get_val();
        csr_val[i]      = 1;
    }

    mat_A.print_matrix("A");

    csr_matrix mat_B;
    mat_B.copy_from(mat_A);

    mat_B.print_matrix("B");

    csr_matrix mat_C;
    mat_C.resize(mat_A.get_m(), mat_B.get_n(), 0);

    mat_A.move_to_device();
    mat_B.move_to_device();
    mat_C.move_to_device();

    auto t1 = std::chrono::high_resolution_clock::now();
    mat_A.multiply_by_matrix(mat_C, mat_B);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << std::endl;

    //mat_C.move_to_host();
    //mat_C.print_matrix("C");*/

    return true;
}