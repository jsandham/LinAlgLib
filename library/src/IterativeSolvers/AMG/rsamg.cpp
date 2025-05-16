//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include "../../../include/IterativeSolvers/AMG/rsamg.h"
#include "../../../include/IterativeSolvers/AMG/amg_aggregation.h"
#include "../../../include/IterativeSolvers/AMG/amg_strength.h"
#include "../../../include/IterativeSolvers/AMG/amg_util.h"
#include "../../../include/IterativeSolvers/slaf.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
#include <chrono>

#include "../../trace.h"

//-------------------------------------------------------------------------------
// function for finding strength matrix size
//-------------------------------------------------------------------------------
static int strength_matrix_size_legacy(const csr_matrix &A, const double theta)
{
    int str_size = 0;
    for (int i = 0; i < A.m; i++)
    {
        int start = A.csr_row_ptr[i];
        int end = A.csr_row_ptr[i + 1];

        double max_value = 0.0;
        for (int j = start; j < end; j++)
        {
            if (-A.csr_val[j] > max_value && i != A.csr_col_ind[j])
            {
                max_value = std::abs(A.csr_val[j]);
            }
        }

        max_value = max_value * theta;
        for (int j = start; j < end; j++)
        {
            if (-A.csr_val[j] > max_value && i != A.csr_col_ind[j])
            {
                str_size++;
            }
        }
    }

    return str_size;
}

//-------------------------------------------------------------------------------
// function for finding strength matrix and lambda array
//-------------------------------------------------------------------------------
static void strength_matrix_legacy(const csr_matrix &A, csr_matrix &strength, std::vector<int> &lambda, const double theta)
{
    strength.csr_row_ptr[0] = 0;

    // determine strength matrix
    int ind = 0;
    for (int i = 0; i < A.m; i++)
    {
        int start = A.csr_row_ptr[i];
        int end = A.csr_row_ptr[i + 1];

        double max_value = 0.0;
        for (int j = start; j < end; j++)
        {
            if (-A.csr_val[j] > max_value && i != A.csr_col_ind[j])
            {
                max_value = fabs(A.csr_val[j]);
            }
        }

        max_value = max_value * theta;

        strength.csr_row_ptr[i + 1] = strength.csr_row_ptr[i];

        for (int j = start; j < end; j++)
        {
            if (-A.csr_val[j] > max_value && i != A.csr_col_ind[j])
            {
                strength.csr_col_ind[ind] = A.csr_col_ind[j];
                lambda[strength.csr_col_ind[ind]]++; // how many strong points are in
                                                     // each column??
                strength.csr_val[ind] = A.csr_val[j];
                ind++;
                strength.csr_row_ptr[i + 1]++;
            }
        }
    }
}

//-------------------------------------------------------------------------------
// structure representing an array
//-------------------------------------------------------------------------------
struct array
{
    int value;
    size_t id;
};

//-------------------------------------------------------------------------------
// compare function for sorting structure array
//-------------------------------------------------------------------------------
static int compare_structs(const void *a, const void *b)
{
    array *struct_a = (array *)a;
    array *struct_b = (array *)b;

    if (struct_a->value < struct_b->value)
        return 1;
    else if (struct_a->value == struct_b->value)
        return 0;
    else
        return -1;
}

//-------------------------------------------------------------------------------
// function for finding c-points and f-points (first pass)
//-------------------------------------------------------------------------------
static void pre_cpoint3_legacy(const csr_matrix &strength, const csr_matrix &strength_transpose, std::vector<int> &lambda,
                        std::vector<unsigned int> &cfpoints)
{
    unsigned locInSortedLambda = 0;
    unsigned numOfNodesToCheck = 0;
    std::vector<unsigned int> nodesToCheck(strength.m, 0);
    std::vector<array> sortedLambda(strength.m);

    // copy lambda into struct array and then sort
    for (size_t i = 0; i < lambda.size(); i++)
    {
        sortedLambda[i].value = lambda[i];
        sortedLambda[i].id = i;
    }

    qsort(sortedLambda.data(), sortedLambda.size(), sizeof(array), compare_structs);

    nodesToCheck[0] = sortedLambda[0].id;
    numOfNodesToCheck++;

    int num_nodes_not_assign = strength.m;
    while (num_nodes_not_assign > 0)
    {
        int max_value = -999;
        unsigned max_index = 0;
        while (locInSortedLambda < strength.m - 1 && lambda[sortedLambda[locInSortedLambda].id] == -999)
        {
            locInSortedLambda++;
        }
        nodesToCheck[0] = sortedLambda[locInSortedLambda].id;

        for (unsigned int i = 0; i < numOfNodesToCheck; i++)
        {
            if (lambda[nodesToCheck[i]] > max_value)
            {
                max_value = lambda[nodesToCheck[i]];
                max_index = nodesToCheck[i];
            }
        }
        numOfNodesToCheck = 1;

        cfpoints[max_index] = 1;
        lambda[max_index] = -999;
        num_nodes_not_assign--;

        // determine how many nonzero entries are in the max_index column of S and
        // what rows those nonzero values are in
        int nnz_in_col = strength_transpose.csr_row_ptr[max_index + 1] - strength_transpose.csr_row_ptr[max_index];

        std::vector<int> index_of_nz(nnz_in_col);
        for (int i = 0; i < nnz_in_col; i++)
        {
            index_of_nz[i] = strength_transpose.csr_col_ind[i + strength_transpose.csr_row_ptr[max_index]];
        }

        // make all connections to cpoint fpoints and update lambda array
        for (int i = 0; i < nnz_in_col; i++)
        {
            if (lambda[index_of_nz[i]] != -999)
            {
                lambda[index_of_nz[i]] = -999;
                num_nodes_not_assign--;
                for (int j = strength.csr_row_ptr[index_of_nz[i]]; j < strength.csr_row_ptr[index_of_nz[i] + 1]; j++)
                {
                    if (lambda[strength.csr_col_ind[j]] != -999)
                    {
                        lambda[strength.csr_col_ind[j]]++;
                        int flag = 0;
                        for (unsigned int k = 0; k < numOfNodesToCheck; k++)
                        {
                            if (nodesToCheck[k] == strength.csr_col_ind[j])
                            {
                                flag = 1;
                                break;
                            }
                        }
                        if (flag == 0)
                        {
                            nodesToCheck[numOfNodesToCheck] = strength.csr_col_ind[j];
                            numOfNodesToCheck++;
                        }
                    }
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------
// function for finding c-points and f-points (second pass)
//-------------------------------------------------------------------------------
static void post_cpoint_legacy(const csr_matrix &strength, std::vector<unsigned int> &cfpoints)
{
    int max_nstrc = 0; // max number of strong connections in any row
    for (int i = 0; i < strength.m; i++)
    {
        int start = strength.csr_row_ptr[i];
        int end = strength.csr_row_ptr[i + 1];

        if (max_nstrc < (end - start))
        {
            max_nstrc = end - start;
        }
    }

    std::vector<int> scpoints(max_nstrc);

    // perform second pass adding c-points where necessary
    for (int i = 0; i < strength.m; i++)
    {
        if (cfpoints[i] == 0) // i is an fpoint
        {
            int start = strength.csr_row_ptr[i];
            int end = strength.csr_row_ptr[i + 1];

            int nstrc = end - start; // number of strong connections in row i
            int scindex = 0;         // number of c-points in row i
            for (int j = start; j < end; j++)
            {
                int col = strength.csr_col_ind[j];
                if (cfpoints[col] == 1)
                {
                    scpoints[scindex] = col;
                    scindex++;
                }
            }

#if (DEBUG)
            if (scindex == 0)
            {
                std::cout << "ERROR: no cpoint for the f-point " << i << std::endl;
            }
#endif

            for (int j = start; j < end; j++)
            {
                int col = strength.csr_col_ind[j];
                if (cfpoints[col] == 0) // col is an fpoint
                {
                    int ind1 = 0, ind2 = 0, flag = 1;
                    while (ind1 < scindex && ind2 < (strength.csr_row_ptr[col + 1] - strength.csr_row_ptr[col]))
                    {
                        if (scpoints[ind1] == strength.csr_col_ind[strength.csr_row_ptr[col] + ind2])
                        {
                            flag = 0;
                            break;
                        }
                        else if (scpoints[ind1] < strength.csr_col_ind[strength.csr_row_ptr[col] + ind2])
                        {
                            ind1++;
                        }
                        else if (scpoints[ind1] > strength.csr_col_ind[strength.csr_row_ptr[col] + ind2])
                        {
                            ind2++;
                        }
                    }

                    if (flag)
                    {
                        cfpoints[col] = 1; // col was an fpoint, but now is a cpoint
                        scpoints[scindex] = col;
                        scindex++;
                    }
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------
// function for finding interpolation weight matrix
//-------------------------------------------------------------------------------
static int weight_matrix_legacy(const csr_matrix &A, const csr_matrix &strength, csr_matrix &W,
                         std::vector<unsigned int> cfpoints)
{
    // determine the number of c-points and f-points
    int cnum = 0;
    int fnum = 0;
    for (int i = 0; i < A.m; i++)
    {
        cnum += cfpoints[i];
    }
    fnum = A.m - cnum;

    // determine the size of the interpolation matrix W
    int wsize = cnum;
    for (int i = 0; i < strength.m; i++)
    {
        int start = strength.csr_row_ptr[i];
        int end = strength.csr_row_ptr[i + 1];

        if (cfpoints[i] == 0)
        {
            for (int j = start; j < end; j++)
            {
                if (cfpoints[strength.csr_col_ind[j]] == 1)
                {
                    wsize++;
                }
            }
        }
    }

    // initialize interpolation matrix W
    W.csr_row_ptr.resize(A.m + 1, 0);
    W.csr_col_ind.resize(wsize, -1);
    W.csr_val.resize(wsize, 0.0);
    W.m = A.m;
    W.n = cnum;
    W.nnz = wsize;

    // modify cfpoints array so that nonzeros now correspond to the cpoint
    // location
    int loc = 0;
    for (size_t i = 0; i < cfpoints.size(); i++)
    {
        if (cfpoints[i] == 1)
        {
            cfpoints[i] = cfpoints[i] + loc;
            loc++;
        }
    }

    // find beta array (sum of weak f-points)
    int ind1 = 0, ind2 = 0, ii = 0;

    std::vector<double> beta(fnum, 0.0);
    for (size_t i = 0; i < cfpoints.size(); i++)
    {
        int A_start = A.csr_row_ptr[i];
        int A_end = A.csr_row_ptr[i + 1];

        int strength_start = strength.csr_row_ptr[i];
        int strength_end = strength.csr_row_ptr[i + 1];

        if (cfpoints[i] == 0)
        {
            ind1 = 0;
            ind2 = 0;
            while (ind1 < (A_end - A_start) && ind2 < (strength_end - strength_start))
            {
                if (A.csr_col_ind[A_start + ind1] == strength.csr_col_ind[strength_start + ind2])
                {
                    ind1++;
                    ind2++;
                }
                else if (A.csr_col_ind[A_start + ind1] < strength.csr_col_ind[strength_start + ind2])
                {
                    if (A.csr_col_ind[A_start + ind1] != i)
                    {
                        beta[ii] = beta[ii] + A.csr_val[A_start + ind1];
                    }
                    ind1++;
                }
            }
            while (ind1 < (A_end - A_start))
            {
                if (A.csr_col_ind[A_start + ind1] != i)
                {
                    beta[ii] = beta[ii] + A.csr_val[A_start + ind1];
                }
                ind1++;
            }
            ii++;
        }
    }

    // find diagonal of A
    std::vector<double> d(A.m, 0.0);
    for (int i = 0; i < A.m; i++)
    {
        int start = A.csr_row_ptr[i];
        int end = A.csr_row_ptr[i + 1];

        for (int j = start; j < end; j++)
        {
            if (i == A.csr_col_ind[j])
            {
                d[i] = A.csr_val[j];
                break;
            }
        }
    }

    // create interpolation matrix W
    double aii = 0.0, aij = 0.0, temp = 0.0;
    int index = 0, rindex = 0;
    ind1 = 0;
    ind2 = 0;
    for (size_t i = 0; i < cfpoints.size(); i++)
    {
        int strength_start = strength.csr_row_ptr[i];
        int strength_end = strength.csr_row_ptr[i + 1];

        if (cfpoints[i] >= 1)
        {
            W.csr_col_ind[index] = ind1;
            W.csr_val[index] = 1.0;
            ind1++;
            index++;
            rindex++;
            W.csr_row_ptr[rindex] = W.csr_row_ptr[rindex - 1] + 1;
        }
        else
        {
            // determine diagonal element a_ii
            aii = d[i];

            // find all strong c-points and f-points in the row i
            int ind3 = 0, ind4 = 0;
            int scnum = 0;
            int sfnum = 0;

            std::vector<int> scpts(strength_end - strength_start, -1);
            std::vector<int> sfpts(strength_end - strength_start, -1);
            std::vector<int> scind(strength_end - strength_start, -1);
            std::vector<double> scval(strength_end - strength_start, 0.0);
            std::vector<double> sfval(strength_end - strength_start, 0.0);

            for (int j = strength_start; j < strength_end; j++)
            {
                int strength_col = strength.csr_col_ind[j];
                double strength_val = strength.csr_val[j];

                if (cfpoints[strength_col] >= 1)
                {
                    scpts[scnum] = strength_col;
                    scval[scnum] = strength_val;
                    scind[scnum] = cfpoints[strength_col] - 1;
                    scnum++;
                }
                else
                {
                    sfpts[sfnum] = strength_col;
                    sfval[sfnum] = strength_val;
                    sfnum++;
                }
            }

#if (DEBUG)
            if (scnum == 0)
            {
                std::cout << "ERROR: no cpoints in row " << i << std::endl;
            }
#endif

            if (sfnum == 0)
            {
                // loop all strong c-points
                for (int k = 0; k < scnum; k++)
                {
                    aij = scval[k];
                    W.csr_col_ind[index] = scind[k];
                    W.csr_val[index] = -(aij) / (aii + beta[ind2]);
                    index++;
                }
            }
            else
            {
                // loop thru all the strong f-points to find alpha array
                std::vector<double> alpha(sfnum, 0.0);
                for (int k = 0; k < sfnum; k++)
                {
                    ind3 = 0;
                    ind4 = 0;
                    while (ind3 < scnum && ind4 < (A.csr_row_ptr[sfpts[k] + 1] - A.csr_row_ptr[sfpts[k]]))
                    {
                        if (scpts[ind3] == A.csr_col_ind[A.csr_row_ptr[sfpts[k]] + ind4])
                        {
                            alpha[k] = alpha[k] + A.csr_val[A.csr_row_ptr[sfpts[k]] + ind4];
                            ind3++;
                            ind4++;
                        }
                        else if (scpts[ind3] < A.csr_col_ind[A.csr_row_ptr[sfpts[k]] + ind4])
                        {
                            ind3++;
                        }
                        else if (scpts[ind3] > A.csr_col_ind[A.csr_row_ptr[sfpts[k]] + ind4])
                        {
                            ind4++;
                        }
                    }
                }

                // loop all strong c-points
                for (int k = 0; k < scnum; k++)
                {
                    aij = scval[k];
                    temp = 0.0;
                    for (int l = 0; l < sfnum; l++)
                    {
                        for (int m = A.csr_row_ptr[sfpts[l]]; m < A.csr_row_ptr[sfpts[l] + 1]; m++)
                        {
                            if (A.csr_col_ind[m] == scpts[k])
                            {
#if (DEBUG)
                                if (alpha[l] == 0.0)
                                {
                                    std::cout << "ERROR: alpha is zero" << std::endl;
                                }
#endif
                                temp = temp + sfval[l] * A.csr_val[m] / alpha[l];
                                break;
                            }
                        }
                    }
                    W.csr_col_ind[index] = scind[k];
                    W.csr_val[index] = -(aij + temp) / (aii + beta[ind2]);
                    index++;
                }
            }
            ind2++;
            rindex++;
            W.csr_row_ptr[rindex] = W.csr_row_ptr[rindex - 1] + scnum;
        }
    }

    return cnum;
}

void rsamg_setup_legacy(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
                 int max_level, heirarchy &hierarchy)
{
    hierarchy.prolongations.resize(max_level);
    hierarchy.restrictions.resize(max_level);
    hierarchy.A_cs.resize(max_level + 1);

    // Set original matrix at level 0 in the hierarchy
    hierarchy.A_cs[0].m = m;
    hierarchy.A_cs[0].n = m;
    hierarchy.A_cs[0].nnz = nnz;
    hierarchy.A_cs[0].csr_row_ptr.resize(m + 1);
    hierarchy.A_cs[0].csr_col_ind.resize(nnz);
    hierarchy.A_cs[0].csr_val.resize(nnz);

    for (int i = 0; i < m + 1; i++)
    {
        hierarchy.A_cs[0].csr_row_ptr[i] = csr_row_ptr[i];
    }

    for (int i = 0; i < nnz; i++)
    {
        hierarchy.A_cs[0].csr_col_ind[i] = csr_col_ind[i];
        hierarchy.A_cs[0].csr_val[i] = csr_val[i];
    }

    double theta = 0.5;

    int level = 0;
    while (level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;

        csr_matrix &A = hierarchy.A_cs[level];
        csr_matrix &A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix &P = hierarchy.prolongations[level];
        csr_matrix &R = hierarchy.restrictions[level];

        // determine size of strength matrix
        int ssize = strength_matrix_size_legacy(A, theta);

        std::cout << "ssize: " << ssize << std::endl;

        csr_matrix strength;
        strength.m = A.m;
        strength.n = A.n;
        strength.nnz = ssize;
        strength.csr_row_ptr.resize(A.m + 1, 0);
        strength.csr_col_ind.resize(ssize, 0);
        strength.csr_val.resize(ssize, 0.0);

        csr_matrix strength_transpose;
        strength_transpose.m = A.m;
        strength_transpose.n = A.n;
        strength_transpose.nnz = ssize;
        strength_transpose.csr_row_ptr.resize(A.m + 1, 0);
        strength_transpose.csr_col_ind.resize(ssize, 0);
        strength_transpose.csr_val.resize(ssize, 0.0);

        std::vector<unsigned int> cfpoints(A.m, 0);
        std::vector<int> lambda(A.m, 0);

        // compute strength matrix S
        strength_matrix_legacy(A, strength, lambda, theta);

        // compute strength transpose matrix S^T
        transpose(strength, strength_transpose);

        // determine c-points and f-points (first pass)
        pre_cpoint3_legacy(strength, strength_transpose, lambda, cfpoints);

        // determine c-points and f-points (second pass)
        post_cpoint_legacy(strength, cfpoints);

        // compute interpolation matrix W
        int numCPoints = weight_matrix_legacy(A, strength, P, cfpoints);

        std::cout << "numCPoints: " << numCPoints << " P.n: " << P.n << std::endl;

        if (P.n == 0)
        {
            break;
        }

        // Compute restriction matrix by transpose of prolongation matrix
        transpose(P, R);

        // Compute coarse grid matrix using Galarkin triple product A_c = R * A * P
        galarkin_triple_product(R, A, P, A_coarse);

        level++;
    }

    hierarchy.total_levels = level;

    std::cout << "Total number of levels in operator hierarchy at the end of the "
                 "setup phase: "
              << level << std::endl;
}































#define F_POINT 0 // F-point
#define C_POINT 1 // C-point

static void direct_interpolation(const csr_matrix &A, const csr_matrix &S, csr_matrix &P, std::vector<uint32_t> cfpoints)
{
    // Determine number of C points. The prolongation operator will have number 
    // of columns equal to the number of C points
    int c_point_count = 0;
    for(int i = 0; i < A.m; i++)
    {
        if(cfpoints[i] == C_POINT)
        {
            c_point_count++;
        }
    }

    P.m = A.m;
    P.n = c_point_count;
    P.csr_row_ptr.resize(P.m + 1);

    // Fill prolongation row pointer array
    P.csr_row_ptr[0] = 0;
    for(int i = 0; i < P.m; i++)
    {
        if(cfpoints[i] == C_POINT)
        {
            P.csr_row_ptr[i + 1] = 1;
        }
        else
        {
            assert(cfpoints[i] == F_POINT);

            int S_row_start = S.csr_row_ptr[i];
            int S_row_end = S.csr_row_ptr[i + 1];

            for(int j = S_row_start; j < S_row_end; j++)
            {
                int S_col_j = S.csr_col_ind[j];

                if(cfpoints[S_col_j] == C_POINT)
                {
                    P.csr_row_ptr[i + 1]++;
                }
            }
        }
    }

    // Exclusive scan on row pointer array
    for(int i = 0; i < P.m; i++)
    {
        P.csr_row_ptr[i + 1] += P.csr_row_ptr[i];
    }

    std::cout << "P.csr_row_ptr" << std::endl;
    for(int i = 0; i < P.m + 1; i++)
    {
        std::cout << P.csr_row_ptr[i] << " ";
    }
    std::cout << "" << std::endl;

    // Allocate prolongation column indices and values arrays
    P.nnz = P.csr_row_ptr[P.m];
    P.csr_col_ind.resize(P.nnz);
    P.csr_val.resize(P.nnz);

    std::cout << "P.nnz: " << P.nnz << std::endl;

    std::vector<uint32_t> cfpoints_ind(A.m + 1);

    cfpoints_ind[0] = 0;
    for(int i = 0; i < P.m; i++)
    {
        cfpoints_ind[i + 1] = cfpoints[i];
    }

    // Exclusive scan on copy of cfpoints array
    for(int i = 0; i < P.m; i++)
    {
        cfpoints_ind[i + 1] += cfpoints_ind[i];
    }

    for(int i = 0; i < P.m; i++)
    {
        int P_row_start = P.csr_row_ptr[i];
        int P_row_end = P.csr_row_ptr[i + 1];

        std::cout << "P_row_start: " << P_row_start << std::endl;

        if(cfpoints[i] == C_POINT)
        {
            assert((P_row_end - P_row_start) == 1);

            P.csr_col_ind[P_row_start] = cfpoints_ind[i];
            P.csr_val[P_row_start] = 1.0;
        }
        else
        {
            double diag = 0.0;
            double offdiag_pos_sum = 0.0;
            double offdiag_neg_sum = 0.0;
            
            int A_row_start = A.csr_row_ptr[i];
            int A_row_end = A.csr_row_ptr[i + 1];

            for(int j = A_row_start; j < A_row_end; j++)
            {
                int A_col_j = A.csr_col_ind[j];

                if(i == A_col_j)
                {
                    diag == A.csr_val[j];
                }
                else
                {
                    if(A.csr_val[j] < 0.0)
                    {
                        offdiag_neg_sum += A.csr_val[j];
                    }
                    else
                    {
                        offdiag_pos_sum += A.csr_val[j];
                    }
                }
            }

            double strong_offdiag_pos_sum = 0.0;
            double strong_offdiag_neg_sum = 0.0;

            int S_row_start = S.csr_row_ptr[i];
            int S_row_end = S.csr_row_ptr[i + 1];

            // S does not contain diagonal
            for(int j = S_row_start; j < S_row_end; j++)
            {
                int A_col_j = A.csr_col_ind[j];

                if(cfpoints[A_col_j] == C_POINT)
                {
                    if(S.csr_val[j] < 0.0)
                    {
                        strong_offdiag_neg_sum += S.csr_val[j];
                    }
                    else
                    {
                        strong_offdiag_pos_sum += S.csr_val[j];
                    }
                }
            }

            std::cout << "offdiag_pos_sum: " << offdiag_pos_sum 
                      << " offdiag_neg_sum: " << offdiag_neg_sum 
                      << " strong_offdiag_pos_sum: " << strong_offdiag_pos_sum 
                      << " strong_offdiag_neg_sum: " << strong_offdiag_neg_sum << std::endl;

            double alpha = offdiag_neg_sum / strong_offdiag_neg_sum;
            double beta  = offdiag_pos_sum / strong_offdiag_pos_sum;

            for(int j = S_row_start; j < S_row_end; j++)
            {
                int S_col_j = S.csr_col_ind[j];

                if(cfpoints[S_col_j] == C_POINT)
                {
                    assert(P_row_start < P_row_end);

                    P.csr_col_ind[P_row_start] = cfpoints_ind[S_col_j];
                    if(S.csr_val[j] < 0.0)
                    {
                        P.csr_val[P_row_start] = -alpha * S.csr_val[j] / diag; 
                    }
                    else
                    {
                        P.csr_val[P_row_start] = -beta * S.csr_val[j] / diag; 
                    }

                    P_row_start++;
                }
            }

            assert(P_row_start == P_row_end);
        }
    }
}

void rsamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
    int max_level, heirarchy &hierarchy)
{
    ROUTINE_TRACE("rsamg_setup");

    auto t1 = std::chrono::high_resolution_clock::now();

    hierarchy.prolongations.resize(max_level);
    hierarchy.restrictions.resize(max_level);
    hierarchy.A_cs.resize(max_level + 1);

    // Set original matrix at level 0 in the hierarchy
    hierarchy.A_cs[0].m = m;
    hierarchy.A_cs[0].n = m;
    hierarchy.A_cs[0].nnz = nnz;
    hierarchy.A_cs[0].csr_row_ptr.resize(m + 1);
    hierarchy.A_cs[0].csr_col_ind.resize(nnz);
    hierarchy.A_cs[0].csr_val.resize(nnz);

    for (int i = 0; i < m + 1; i++)
    {
        hierarchy.A_cs[0].csr_row_ptr[i] = csr_row_ptr[i];
    }

    for (int i = 0; i < nnz; i++)
    {
        hierarchy.A_cs[0].csr_col_ind[i] = csr_col_ind[i];
        hierarchy.A_cs[0].csr_val[i] = csr_val[i];
    }

    double theta = 0.2;

    int level = 0;
    while (level < max_level)
    {
        std::cout << "Compute operators at coarse level: " << level << std::endl;

        csr_matrix &A = hierarchy.A_cs[level];
        csr_matrix &A_coarse = hierarchy.A_cs[level + 1];
        csr_matrix &P = hierarchy.prolongations[level];
        csr_matrix &R = hierarchy.restrictions[level];

        csr_matrix S;
        csr_matrix ST;

        std::vector<int> connections(A.nnz, 0);
        std::vector<uint32_t> cfpoints(A.m, 0);

        compute_classical_strong_connections(A, theta, S, connections);

        transpose(S, ST);

        compute_cfpoint_first_pass(S, ST, cfpoints);

        compute_cfpoint_second_pass(S, cfpoints);

        std::cout << "cfpoints (second pass)" << std::endl;
        for(size_t i = 0; i < cfpoints.size(); i++)
        {
            std::cout << cfpoints[i] << " ";
        }
        std::cout << "" << std::endl;

        print_matrix("S", S.csr_row_ptr.data(), S.csr_col_ind.data(), S.csr_val.data(), S.m, S.n, S.nnz);

        // compute interpolation matrix P
        direct_interpolation(A, S, P, cfpoints);

        print_matrix("Weight", P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), P.m, P.n, P.nnz);

        if (P.n == 0)
        {
            break;
        }

        // Compute restriction matrix by transpose of prolongation matrix
        transpose(P, R);

        // Compute coarse grid matrix using Galarkin triple product A_c = R * A * P
        galarkin_triple_product(R, A, P, A_coarse);

        level++;
    }

    hierarchy.total_levels = level;

    std::cout << "Total number of levels in operator hierarchy at the end of the "
                 "setup phase: "
              << level << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Ruge Steuben AMG setup time: " << ms_double.count() << "ms" << std::endl;
}