#ifndef TEST_ITERATIVE_H__
#define TEST_ITERATIVE_H__

#include <string>

namespace Testing
{
bool test_jacobi(const std::string &matrix_file);
bool test_gauss_seidel(const std::string &matrix_file);
bool test_sor(const std::string &matrix_file);
bool test_symmetric_gauss_seidel(const std::string &matrix_file);
bool test_ssor(const std::string &matrix_file);
} // namespace Testing

#endif