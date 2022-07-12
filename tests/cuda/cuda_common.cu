#include "cuda_common.cuh"
#include <iomanip>

// cudaError related

bool CudaErrorMatcher::match(cudaError const &in) const {
    return in == expect_code;
}

std::string CudaErrorMatcher::describe() const {
    return std::string(cudaGetErrorName(expect_code));
}

namespace Catch {

std::string StringMaker<cudaError>::convert(cudaError const &code) {
    return std::string() + "{" + cudaGetErrorName(code) +
           "}: " + cudaGetErrorString(code);
}

} // namespace Catch
