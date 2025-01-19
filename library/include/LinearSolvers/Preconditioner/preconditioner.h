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

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <vector>

class preconditioner
{
    public:
        preconditioner(){};
        virtual ~preconditioner(){};

        virtual void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) = 0;
        virtual void solve(const double* rhs, double* x, int n) const = 0;
};

class jacobi_precond : public preconditioner
{
    private:
        std::vector<double> diag;

    public:
        jacobi_precond();
        ~jacobi_precond();

        void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;
        void solve(const double* rhs, double* x, int n) const override;
};

class ilu_precond : public preconditioner
{
    private:
        std::vector<int> csr_row_ptr_LU;
        std::vector<int> csr_col_ind_LU;
        std::vector<double> csr_val_LU;

    public:
        ilu_precond();
        ~ilu_precond();

        void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;
        void solve(const double* rhs, double* x, int n) const override;
};

class ic_precond : public preconditioner
{
    private:
        std::vector<int> csr_row_ptr_LLT;
        std::vector<int> csr_col_ind_LLT;
        std::vector<double> csr_val_LLT;

    public:
        ic_precond();
        ~ic_precond();

        void build(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz) override;
        void solve(const double* rhs, double* x, int n) const override;
};

#endif