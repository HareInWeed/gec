#include <gec/utils/cuda_utils.cuh>

#include "common.hpp"
#include "field.hpp"

#include "cuda_common.cuh"

#include "configured_catch.hpp"

using namespace gec;
using namespace utils;

__global__ void test_set_get_cc_cf_(bool *flags) {
    set_cc_cf_(true);
    flags[0] = get_cc_cf_();
    flags[0] = get_cc_cf_();
    set_cc_cf_(false);
    flags[1] = get_cc_cf_();
    flags[1] = get_cc_cf_();
    set_cc_cf_(true);
    set_cc_cf_(false);
    flags[2] = get_cc_cf_();
    flags[2] = get_cc_cf_();
    set_cc_cf_(false);
    set_cc_cf_(true);
    flags[3] = get_cc_cf_();
    flags[3] = get_cc_cf_();
};
TEST_CASE("cuda CC.CF flag", "[cuda][intrinsics]") {
    bool *buf, *d_buf;
    size_t buf_size = sizeof(bool) * 4;
    CUDA_REQUIRE(cudaMallocHost(&buf, buf_size));
    CUDA_REQUIRE(cudaMalloc(&d_buf, buf_size));

    test_set_get_cc_cf_<<<1, 1>>>(d_buf);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(buf, d_buf, buf_size, cudaMemcpyDeviceToHost));

    REQUIRE(buf[0]);
    REQUIRE(!buf[1]);
    REQUIRE(!buf[2]);
    REQUIRE(buf[3]);

    CUDA_REQUIRE(cudaFreeHost(buf));
    CUDA_REQUIRE(cudaFree(d_buf));
}

__global__ void test_add_cc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    add_cc_<uint32_t>(vals[i], (a), (b));                                      \
    carries[i] = get_cc_cf_();                                                 \
    ++i

    test_helper(false, 0, 0);
    test_helper(true, 0, 0);
    test_helper(false, 1, 0);
    test_helper(true, 1, 0);
    test_helper(false, 0xffff0000, 0xffff);
    test_helper(true, 0xffff0000, 0xffff);
    test_helper(false, 0xffff0000, 0x10000);
    test_helper(true, 0xffff0000, 0x10000);

#undef test_helper
};
TEST_CASE("add_cc_", "[cuda][intrinsics]") {
    constexpr size_t N = 8;

    using T = uint32_t;
    T *vals, *d_vals;
    size_t vals_size = sizeof(T) * N;
    bool *carries, *d_carries;
    size_t carries_size = sizeof(bool) * N;
    CUDA_REQUIRE(cudaMallocHost(&vals, vals_size));
    CUDA_REQUIRE(cudaMalloc(&d_vals, vals_size));
    CUDA_REQUIRE(cudaMallocHost(&carries, carries_size));
    CUDA_REQUIRE(cudaMalloc(&d_carries, carries_size));

    test_add_cc_<<<1, 1>>>(d_vals, d_carries);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(vals, d_vals, vals_size, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpy(carries, d_carries, carries_size, cudaMemcpyDeviceToHost));

    int i = 0;

#define test_helper(carry, s)                                                  \
    REQUIRE(s == vals[i]);                                                     \
    REQUIRE(carry == carries[i]);                                              \
    ++i

    test_helper(false, 0);
    test_helper(false, 0);
    test_helper(false, 1);
    test_helper(false, 1);
    test_helper(false, 0xffffffff);
    test_helper(false, 0xffffffff);
    test_helper(true, 0x0);
    test_helper(true, 0x0);
#undef test_helper

    CUDA_REQUIRE(cudaFreeHost(vals));
    CUDA_REQUIRE(cudaFree(d_vals));
    CUDA_REQUIRE(cudaFreeHost(carries));
    CUDA_REQUIRE(cudaFree(d_carries));
}

__global__ void test_addc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    addc_<uint32_t>(vals[i], (a), (b));                                        \
    carries[i] = get_cc_cf_();                                                 \
    ++i

    test_helper(false, 0, 0);
    test_helper(true, 0, 0);
    test_helper(false, 1, 0);
    test_helper(true, 1, 0);
    test_helper(false, 0xffff0000, 0xffff);
    test_helper(true, 0xffff0000, 0xffff);
    test_helper(false, 0xffff0000, 0x10000);
    test_helper(true, 0xffff0000, 0x10000);

#undef test_helper
};
TEST_CASE("addc_", "[cuda][intrinsics]") {
    constexpr size_t N = 8;

    using T = uint32_t;
    T *vals, *d_vals;
    size_t vals_size = sizeof(T) * N;
    bool *carries, *d_carries;
    size_t carries_size = sizeof(bool) * N;
    CUDA_REQUIRE(cudaMallocHost(&vals, vals_size));
    CUDA_REQUIRE(cudaMalloc(&d_vals, vals_size));
    CUDA_REQUIRE(cudaMallocHost(&carries, carries_size));
    CUDA_REQUIRE(cudaMalloc(&d_carries, carries_size));

    test_addc_<<<1, 1>>>(d_vals, d_carries);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(vals, d_vals, vals_size, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpy(carries, d_carries, carries_size, cudaMemcpyDeviceToHost));

    int i = 0;

#define test_helper(carry, s)                                                  \
    REQUIRE(s == vals[i]);                                                     \
    REQUIRE(carry == carries[i]);                                              \
    ++i

    test_helper(false, 0);
    test_helper(true, 1);
    test_helper(false, 1);
    test_helper(true, 2);
    test_helper(false, 0xffffffff);
    test_helper(true, 0x0);
    test_helper(false, 0x0);
    test_helper(true, 0x1);

#undef test_helper

    CUDA_REQUIRE(cudaFreeHost(vals));
    CUDA_REQUIRE(cudaFree(d_vals));
    CUDA_REQUIRE(cudaFreeHost(carries));
    CUDA_REQUIRE(cudaFree(d_carries));
}

__global__ void test_addc_cc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    addc_cc_<uint32_t>(vals[i], (a), (b));                                     \
    carries[i] = get_cc_cf_();                                                 \
    ++i

    test_helper(false, 0, 0);
    test_helper(true, 0, 0);
    test_helper(false, 1, 0);
    test_helper(true, 1, 0);
    test_helper(false, 0xffff0000, 0xffff);
    test_helper(true, 0xffff0000, 0xffff);
    test_helper(false, 0xffff0000, 0x10000);
    test_helper(true, 0xffff0000, 0x10000);

#undef test_helper
};
TEST_CASE("addc_cc_", "[cuda][intrinsics]") {
    constexpr size_t N = 8;

    using T = uint32_t;
    T *vals, *d_vals;
    size_t vals_size = sizeof(T) * N;
    bool *carries, *d_carries;
    size_t carries_size = sizeof(bool) * N;
    CUDA_REQUIRE(cudaMallocHost(&vals, vals_size));
    CUDA_REQUIRE(cudaMalloc(&d_vals, vals_size));
    CUDA_REQUIRE(cudaMallocHost(&carries, carries_size));
    CUDA_REQUIRE(cudaMalloc(&d_carries, carries_size));

    test_addc_cc_<<<1, 1>>>(d_vals, d_carries);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(vals, d_vals, vals_size, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpy(carries, d_carries, carries_size, cudaMemcpyDeviceToHost));

    int i = 0;

#define test_helper(carry, s)                                                  \
    REQUIRE(s == vals[i]);                                                     \
    REQUIRE(carry == carries[i]);                                              \
    ++i

    test_helper(false, 0);
    test_helper(false, 1);
    test_helper(false, 1);
    test_helper(false, 2);
    test_helper(false, 0xffffffff);
    test_helper(true, 0x0);
    test_helper(true, 0x0);
    test_helper(true, 0x1);

#undef test_helper

    CUDA_REQUIRE(cudaFreeHost(vals));
    CUDA_REQUIRE(cudaFree(d_vals));
    CUDA_REQUIRE(cudaFreeHost(carries));
    CUDA_REQUIRE(cudaFree(d_carries));
}

def_array(SmallMod, LIMB_T, 3, 0xb, 0x0, 0x7);

template <typename Int, typename Rng>
__global__ void test_cuda_sampling_kernel(size_t seed, Int *x,
                                          GecRng<Rng> *rng_p) {
    auto &rng = *rng_p;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &rng.get_rng());

    typename Int::template Context<> ctx;

    Int::sample(x[0], rng);

    Int::sample_non_zero(x[1], rng);

    Int::sample(x[2], x[1], rng);

    Int::sample(x[3], x[2], x[1], rng, ctx);

    Int::sample_inclusive(x[4], x[1], rng);

    Int::sample_inclusive(x[5], x[2], x[1], rng, ctx);
}

template <typename Int, typename Rng>
void test_cuda_sampling() {
    std::random_device rd;
    Int *x, *d_x;
    size_t x_size = 6 * sizeof(Int);
    GecRng<Rng> *d_rng;
    CUDA_REQUIRE(cudaMallocHost(&x, x_size));
    CUDA_REQUIRE(cudaMalloc(&d_x, x_size));
    CUDA_REQUIRE(cudaMalloc(&d_rng, sizeof(GecRng<Rng>)));

    test_cuda_sampling_kernel<Int, Rng><<<1, 1>>>(rd(), d_x, d_rng);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    CUDA_REQUIRE(cudaMemcpy(x, d_x, x_size, cudaMemcpyDeviceToHost));
    REQUIRE(x[0] < Int::mod());
    REQUIRE(!x[1].is_zero());
    REQUIRE(x[1] < Int::mod());
    REQUIRE(x[2] < x[1]);
    REQUIRE(x[3] < x[1]);
    REQUIRE(x[2] <= x[3]);
    REQUIRE(x[4] <= x[1]);
    REQUIRE(x[5] <= x[1]);
    REQUIRE(x[2] <= x[5]);

    CUDA_REQUIRE(cudaFreeHost(x));
    CUDA_REQUIRE(cudaFree(d_x));
    CUDA_REQUIRE(cudaFree(d_rng));
}

TEST_CASE("cuda random sampling", "[add_group][field][random][cuda]") {
    using F1 = Field160;
    using F2 = Field160_2;
    using G = ADD_GROUP(LIMB_T, 3, 0, SmallMod);

    test_cuda_sampling<F1, curandStateXORWOW_t>();
    // test_cuda_sampling<F2, curandStateSobol64_t>();
    test_cuda_sampling<G, curandStateXORWOW_t>();
}
