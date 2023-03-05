#include <configured_catch.hpp>
#include <gec/curve/secp256k1.hpp>
#include <iomanip>
#include <utility>
#include <vector>

class CudaErrorMatcher : public Catch::MatcherBase<cudaError> {
    cudaError expect_code;

  public:
    CudaErrorMatcher() : expect_code(cudaSuccess) {}
    CudaErrorMatcher(cudaError expect_code) : expect_code(expect_code) {}

    bool match(cudaError const &in) const override { return in == expect_code; }
    std::string describe() const override {
        return std::string(cudaGetErrorName(expect_code));
    }
};

namespace Catch {

template <>
struct StringMaker<cudaError> {
    static std::string convert(cudaError const &code) {
        return std::string() + "{" + cudaGetErrorName(code) +
               "}: " + cudaGetErrorString(code);
    }
};

} // namespace Catch

#define CUDA_REQUIRE(code) REQUIRE(cudaSuccess == (code))

using namespace gec;
using S = curve::secp256k1::Scalar;
using C = curve::secp256k1::Curve<curve::ProjectiveCurve>;
using curve::secp256k1::Gen;

__global__ static void point_add_kernel(C *sum, C *p1, C *p2, size_t n) {
    const size_t steps = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    C tsum, tp1, tp2;
    while (idx < n) {
        tp1 = p1[idx];
        tp2 = p2[idx];
        C::add(tsum, tp1, tp2);
        sum[idx] = tsum;
        idx += steps;
    }
}

__global__ static void point_mul_kernel(C *prod, S *s, C *p, size_t n) {
    const size_t steps = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // C tp, tprod;
    // S ts;
    while (idx < n) {
        // ts = s[idx];
        // tp = p[idx];
        // C::mul(tprod, ts, tp);
        // prod[idx] = tprod;
        C::mul(prod[idx], s[idx], p[idx]);
        idx += steps;
    }
}

TEST_CASE("gec_cuda", "[gec][bench]") {
    std::random_device rd;
    auto seed = rd();
    auto rng = make_gec_rng(std::mt19937(seed));

    const size_t factor = 16;
    int gridSize1, blockSize1;
    cudaOccupancyMaxPotentialBlockSize(&gridSize1, &blockSize1,
                                       point_add_kernel);
    int gridSize2, blockSize2;
    cudaOccupancyMaxPotentialBlockSize(&gridSize2, &blockSize2,
                                       point_mul_kernel);
    std::stringstream bench_name;

    const size_t n1 = factor * gridSize1 * blockSize1;
    const size_t n2 = factor * gridSize2 * blockSize2;
    const size_t n = std::max(n1, n2);

    const C G{Gen.x(), Gen.y(), Gen.z()};

    std::vector<S> s1(n), s2(n);
    std::vector<C> p1(n), p2(n);

    S *d_s;
    C *d_p1, *d_p2, *d_p3;
    CUDA_REQUIRE(cudaMalloc(&d_s, n * sizeof(S)));
    CUDA_REQUIRE(cudaMalloc(&d_p1, n * sizeof(C)));
    CUDA_REQUIRE(cudaMalloc(&d_p2, n * sizeof(C)));
    CUDA_REQUIRE(cudaMalloc(&d_p3, n * sizeof(C)));

    for (size_t k = 0; k < n1; ++k) {
        S::sample(s1[k], rng);
        C::mul(p1[k], s1[k], G);
        S::sample(s2[k], rng);
        C::mul(p2[k], s2[k], G);
    }
    CUDA_REQUIRE(
        cudaMemcpy(d_p1, p1.data(), n1 * sizeof(C), cudaMemcpyHostToDevice));
    CUDA_REQUIRE(
        cudaMemcpy(d_p2, p2.data(), n1 * sizeof(C), cudaMemcpyHostToDevice));

    bench_name.str("");
    bench_name << "secp256k1 point add x" << n1;
    BENCHMARK_ADVANCED(bench_name.str())
    (Catch::Benchmark::Chronometer meter) {
        cudaError_t sync_result;
        meter.measure([&]() {
            point_add_kernel<<<gridSize1, blockSize1>>>(d_p3, d_p1, d_p2, n1);
            return sync_result = cudaDeviceSynchronize();
        });
        CUDA_REQUIRE(sync_result);
        CUDA_REQUIRE(cudaGetLastError());
    };

    for (size_t k = 0; k < n2; ++k) {
        p1[k] = G;
    }
    CUDA_REQUIRE(
        cudaMemcpy(d_p1, p1.data(), n2 * sizeof(C), cudaMemcpyHostToDevice));
    for (size_t k = 0; k < n2; ++k) {
        S::sample(s1[k], rng);
    }
    CUDA_REQUIRE(
        cudaMemcpy(d_s, s1.data(), n2 * sizeof(S), cudaMemcpyHostToDevice));

    bench_name.str("");
    bench_name << "secp256k1 scalar mul x" << n2;
    BENCHMARK_ADVANCED(bench_name.str())
    (Catch::Benchmark::Chronometer meter) {
        cudaError_t sync_result;
        meter.measure([&]() {
            point_mul_kernel<<<gridSize2, blockSize2>>>(d_p2, d_s, d_p1, n2);
            return sync_result = cudaDeviceSynchronize();
        });
        CUDA_REQUIRE(sync_result);
        CUDA_REQUIRE(cudaGetLastError());
    };

    CUDA_REQUIRE(cudaFree(d_s));
    CUDA_REQUIRE(cudaFree(d_p1));
    CUDA_REQUIRE(cudaFree(d_p2));
    CUDA_REQUIRE(cudaFree(d_p3));
}