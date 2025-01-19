//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#ifndef RSAMG_OLD_H
#define RSAMG_OLD_H

/*! \file
 *  \brief raamg_old.h provides interface for classical algebraic multigrid
 * solver
 */

void amg(const int r[], const int c[], const double v[], double x[], const double b[], const int n, const double theta,
         const double tol);

void amg_solve(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], double *wv[], double x[],
               const double b[], int ASizes[], int n1, int n2, int level, int count);

void amg_init(const int r[], const int c[], const double v[], int *ar[], int *ac[], double *av[], double *ad[],
              int ASizes[], const int n);

int amg_setup(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], double *wv[], int ASizes[],
              int level, const double theta);

int strength_matrix_size(const int r[], const int c[], const double v[], int rptr_size, const double theta);

void strength_matrix(const int r[], const int c[], const double v[], int sr[], int sc[], double sv[], int lambda[],
                     int rptr_size, const double theta);

void strength_transpose_matrix(int sr[], int sc[], double sv[], int str[], int stc[], double stv[], int lambda[],
                               int rptr_size, const double theta);

void pre_cpoint(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], int rptr_size);

void pre_cpoint3(int sr[], int sc[], int str[], int stc[], int lambda[], unsigned cfpoints[], int rptr_size);

void post_cpoint(int sr[], int sc[], unsigned cfpoints[], int rptr_size);

int weight_matrix(const int r[], const int c[], const double v[], double d[], int sr[], int sc[], double sv[],
                  int *wr[], int *wc[], double *wv[], unsigned cfpoints[], int rptr_size, int level);

void galerkin_prod(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], int rptr_size, int m,
                   int level);

void galerkin_prod2(int *ar[], int *ac[], double *av[], double *ad[], int *wr[], int *wc[], double *wv[], int rptr_size,
                    int m, int level);

void galerkin_prod3(int *ar[], int *ac[], double *av[], int *wr[], int *wc[], double *wv[], int rptr_size, int m,
                    int level);

void sort(int array1[], double array2[], int start, int end);

int compare_structs(const void *a, const void *b);

#endif
