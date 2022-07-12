#include <configured_catch.hpp>

class CudaErrorMatcher : public Catch::MatcherBase<cudaError> {
    cudaError expect_code;

  public:
    CudaErrorMatcher() : expect_code(cudaSuccess) {}
    CudaErrorMatcher(cudaError expect_code) : expect_code(expect_code) {}

    bool match(cudaError const &in) const override;
    std::string describe() const override;
};

namespace Catch {

template <>
struct StringMaker<cudaError> {
    static std::string convert(cudaError const &code);
};

} // namespace Catch

#define CUDA_REQUIRE(code) REQUIRE(cudaSuccess == (code))
