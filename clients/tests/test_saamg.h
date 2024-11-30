#ifndef TEST_SAAMG_H__
#define TEST_SAAMG_H__

#include <string>
#include "linalg.h"

namespace Testing
{
    bool test_saamg(const std::string& matrix_file, int presmoothing, int postsmoothing, Cycle cycle, Smoother smoother);
}

#endif