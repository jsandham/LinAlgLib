#include "test_saamg.h"
#include "utility.h"

#include <iostream>
#include <cmath>

#include "linalg.h"

bool Testing::test_saamg(const std::string& matrix_file)
{
    std::vector<int> csr_row_ptr;
	std::vector<int> csr_col_ind;
	std::vector<double> csr_val;
	load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val);

	int m = (int)csr_row_ptr.size() - 1;
	int n = m;
	int nnz = (int)csr_val.size();

	// Solution vector
	std::vector<double> x(m, 0.0);

	// Righthand side vector
	std::vector<double> b(m, 1.0);

	heirarchy hierachy;
	saamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 100, hierachy);

	amg_solve(hierachy, x.data(), b.data(), 10, 10, 0.00001, Cycle::Vcycle, Smoother::Gauss_Siedel);
	//amg_solve(hierachy, x.data(), b.data(), 2, 2, 0.00001, Cycle::Wcycle, Smoother::Gauss_Siedel);
	//amg_solve(hierachy, x.data(), b.data(), 2, 2, 0.00001, Cycle::Wcycle, Smoother::SOR);

    std::vector<double> residual(m, 0.0);
    for (int i = 0; i < m; i++)
	{
		int row_begin = csr_row_ptr[i];
		int row_end = csr_row_ptr[i + 1];

		double sum = 0;
		for (int j = row_begin; j < row_end; j++)
		{
			sum += csr_val[j] * x[csr_col_ind[j]];
		}

		residual[i] = sum - b[i];
	}

    bool solution_valid = true;
    for(size_t i = 0; i < x.size(); i++)
    {
        if(std::isnan(x[i]) || std::isinf(x[i]))
        {
            solution_valid = false;
            break;
        }
    }

    double max_error = 0.0;
    for(size_t i = 0; i < residual.size(); i++)
    {
        max_error = std::max(max_error, std::abs(residual[i]));
    }

    std::cout << "max_error: " << max_error << std::endl;

    // std::cout << "residual" << std::endl;
    // for(size_t i = 0; i < std::min((size_t)10, residual.size()); i++)
    // {
    //    std::cout << residual[i] << " " << x[i] << " ";
    // }
    // std::cout << "" << std::endl;

    return (max_error < 0.00001 && solution_valid);
}