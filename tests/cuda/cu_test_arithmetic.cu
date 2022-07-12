#include <common.hpp>
#include <utils.hpp>

#include <gec/utils/arithmetic.hpp>

#include <limits>
#include <random>

#include "cuda_common.cuh"
#include <configured_catch.hpp>

using namespace gec;

template <typename Int>
__global__ static void test_uint_add_with_carry_kernel(bool *carry, Int *xs) {
    bool c = *carry;
    Int x1 = xs[0], x2 = xs[1], sum;
    *carry = utils::uint_add_with_carry(sum, x1, x2, c);
    xs[0] = sum;
}
template <typename Int>
static void test_uint_add_with_carry(std::random_device::result_type seed) {
    CAPTURE(seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Int> gen_i;
    std::uniform_int_distribution<int> gen_b;

    const size_t bytes = sizeof(Int) * 2;

    Int x1 = gen_i(rng), x2 = gen_i(rng), sum;
    bool c = gen_b(rng) >= 0;

    Int r_xs[2] = {x1, x2};
    bool r_c = c;
    Int *d_xs;
    bool *d_c;

    CUDA_REQUIRE(cudaMalloc(&d_xs, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_c, sizeof(bool)));

    CUDA_REQUIRE(cudaMemcpyAsync(d_xs, r_xs, bytes, cudaMemcpyHostToDevice));
    CUDA_REQUIRE(
        cudaMemcpyAsync(d_c, &r_c, sizeof(bool), cudaMemcpyHostToDevice));
    test_uint_add_with_carry_kernel<Int><<<1, 1>>>(d_c, d_xs);
    CUDA_REQUIRE(cudaMemcpyAsync(r_xs, d_xs, bytes, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_c, d_c, sizeof(bool), cudaMemcpyDeviceToHost));

    c = utils::uint_add_with_carry(sum, x1, x2, c);

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(c == r_c);
    REQUIRE(r_xs[0] == sum);

    CUDA_REQUIRE(cudaFree(d_xs));
    CUDA_REQUIRE(cudaFree(d_c));
}
TEST_CASE("cuda uint_add_with_carry", "[arithmetic][cuda]") {
    std::random_device rd;
    test_uint_add_with_carry<uint32_t>(rd());
    test_uint_add_with_carry<uint64_t>(rd());
}

template <typename Int>
__global__ static void test_uint_mul_lh_kernel(Int *xs) {
    Int x1 = xs[0], x2 = xs[1], l, h;
    utils::uint_mul_lh(l, h, x1, x2);
    xs[0] = l;
    xs[1] = h;
}
template <typename Int>
static void test_uint_mul_lh(std::random_device::result_type seed) {
    CAPTURE(seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Int> gen_i;

    const size_t bytes = sizeof(Int) * 2;

    Int x1 = gen_i(rng), x2 = gen_i(rng), l, h;

    Int r_xs[2] = {x1, x2};
    Int *d_xs;

    CUDA_REQUIRE(cudaMalloc(&d_xs, bytes));

    CUDA_REQUIRE(cudaMemcpyAsync(d_xs, r_xs, bytes, cudaMemcpyHostToDevice));
    test_uint_mul_lh_kernel<Int><<<1, 1>>>(d_xs);
    CUDA_REQUIRE(cudaMemcpyAsync(r_xs, d_xs, bytes, cudaMemcpyDeviceToHost));

    utils::uint_mul_lh(l, h, x1, x2);

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(l == r_xs[0]);
    REQUIRE(h == r_xs[1]);

    CUDA_REQUIRE(cudaFree(d_xs));
}
TEST_CASE("cuda uint_mul_lh", "[arithmetic][cuda]") {
    std::random_device rd;
    test_uint_mul_lh<uint32_t>(rd());
    test_uint_mul_lh<uint64_t>(rd());
}

template <size_t N, typename Int>
__global__ static void test_seq_add_mul_limb_kernel(Int *a, Int *b, Int *x) {
    *x = utils::seq_add_mul_limb<N>(a, b, *x);
}
template <typename Int>
static void test_seq_add_mul_limb(std::random_device::result_type seed) {
    CAPTURE(seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Int> gen_i;

    constexpr size_t N = 5;

    Int a[N], b[N];
    for (size_t k = 0; k < N; ++k) {
        a[k] = gen_i(rng);
        b[k] = gen_i(rng);
    }
    Int x = gen_i(rng);

    const size_t bytes = sizeof(Int) * N;

    Int r_a[N], r_x;
    Int *d_a, *d_b, *d_x;

    CUDA_REQUIRE(cudaMalloc(&d_a, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_b, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_x, sizeof(Int)));

    CUDA_REQUIRE(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_x, &x, sizeof(Int), cudaMemcpyHostToDevice));
    test_seq_add_mul_limb_kernel<N><<<1, 1>>>(d_a, d_b, d_x);
    CUDA_REQUIRE(cudaMemcpyAsync(r_a, d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_x, d_x, sizeof(Int), cudaMemcpyDeviceToHost));

    x = utils::seq_add_mul_limb<N>(a, b, x);

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    for (size_t k = 0; k < N; ++k) {
        CAPTURE(k);
        REQUIRE(a[k] == r_a[k]);
    }
    REQUIRE(x == r_x);

    CUDA_REQUIRE(cudaFree(d_a));
    CUDA_REQUIRE(cudaFree(d_b));
    CUDA_REQUIRE(cudaFree(d_x));
}
TEST_CASE("cuda seq_add_mul_limb", "[arithmetic][cuda]") {
    std::random_device rd;
    test_seq_add_mul_limb<uint32_t>(rd());
    test_seq_add_mul_limb<uint64_t>(rd());
}
