#include <common.hpp>

#include <gec/bigint.hpp>

#include "cuda_common.cuh"
#include <configured_catch.hpp>

#include <thrust/random.h>

using namespace gec;
using namespace bigint;

class GEC_EMPTY_BASES Int160 : public ArrayBE<LIMB_T, LN_160>,
                               public BigintMixin<Int160, LIMB_T, LN_160> {
  public:
    using ArrayBE::ArrayBE;
};

class GEC_EMPTY_BASES Int160_2
    : public ArrayBE<LIMB2_T, LN2_160>,
      public BigintMixin<Int160_2, LIMB2_T, LN2_160> {
  public:
    using ArrayBE::ArrayBE;
};

template <typename Int>
__global__ static void test_cuda_bigint_constructor_kernel(Int *xs) {
    new (xs) Int;
    new (xs + 1) Int(0x1234);
    new (xs + 2) Int(1, 2, 3, 4, 5);
}
template <typename Int>
static void test_cuda_bigint_constructor() {
    const size_t bytes = sizeof(Int) * 3;

    Int x1, x2(0x1234), x3(1, 2, 3, 4, 5);
    Int xs[3];
    Int *d_xs;

    CUDA_REQUIRE(cudaMalloc(&d_xs, bytes));
    test_cuda_bigint_constructor_kernel<Int><<<1, 1>>>(d_xs);
    CUDA_REQUIRE(cudaMemcpyAsync(&xs, d_xs, bytes, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(x1.array()[0] == xs[0].array()[0]);
    REQUIRE(x1.array()[1] == xs[0].array()[1]);
    REQUIRE(x1.array()[2] == xs[0].array()[2]);
    REQUIRE(x1.array()[3] == xs[0].array()[3]);
    REQUIRE(x1.array()[4] == xs[0].array()[4]);

    REQUIRE(x2.array()[0] == xs[1].array()[0]);
    REQUIRE(x2.array()[1] == xs[1].array()[1]);
    REQUIRE(x2.array()[2] == xs[1].array()[2]);
    REQUIRE(x2.array()[3] == xs[1].array()[3]);
    REQUIRE(x2.array()[4] == xs[1].array()[4]);

    REQUIRE(x3.array()[0] == xs[2].array()[0]);
    REQUIRE(x3.array()[1] == xs[2].array()[1]);
    REQUIRE(x3.array()[2] == xs[2].array()[2]);
    REQUIRE(x3.array()[3] == xs[2].array()[3]);
    REQUIRE(x3.array()[4] == xs[2].array()[4]);

    CUDA_REQUIRE(cudaFree(d_xs));
}
TEST_CASE("cuda bigint constructor", "[bigint][cuda]") {
    test_cuda_bigint_constructor<Int160>();
}

template <typename Int>
__global__ static void test_cuda_bigint_comparison_kernel(bool *res, Int *xs1,
                                                          Int *xs2) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    res[6 * id + 0] = xs1[id] == xs2[id];
    res[6 * id + 1] = xs1[id] != xs2[id];
    res[6 * id + 2] = xs1[id] < xs2[id];
    res[6 * id + 3] = xs1[id] <= xs2[id];
    res[6 * id + 4] = xs1[id] > xs2[id];
    res[6 * id + 5] = xs1[id] >= xs2[id];
}
template <typename Int>
static void test_cuda_bigint_comparison(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    constexpr int BN = 6;
    constexpr int TN = 64;
    constexpr int N = BN * TN;
    constexpr size_t bytes = sizeof(Int) * N;

    Int xs1[N], xs2[N];
    bool res[6 * N];
    for (size_t k = 0; k < N; ++k) {
        Int::sample(xs1[k], rng);
        Int::sample(xs2[k], rng);
    }

    Int *d_xs1, *d_xs2;
    bool *d_res;

    CUDA_REQUIRE(cudaMalloc(&d_xs1, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_xs2, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_res, sizeof(bool) * 6 * N));
    CUDA_REQUIRE(cudaMemcpyAsync(d_xs1, xs1, bytes, cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_xs2, xs2, bytes, cudaMemcpyHostToDevice));
    test_cuda_bigint_comparison_kernel<Int><<<BN, TN>>>(d_res, d_xs1, d_xs2);
    CUDA_REQUIRE(cudaMemcpyAsync(res, d_res, sizeof(bool) * 6 * N,
                                 cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    for (int k = 0; k < N; ++k) {
        CAPTURE(k, xs1[k], xs2[k]);
        REQUIRE((xs1[k] == xs2[k]) == res[6 * k + 0]);
        REQUIRE((xs1[k] != xs2[k]) == res[6 * k + 1]);
        REQUIRE((xs1[k] < xs2[k]) == res[6 * k + 2]);
        REQUIRE((xs1[k] <= xs2[k]) == res[6 * k + 3]);
        REQUIRE((xs1[k] > xs2[k]) == res[6 * k + 4]);
        REQUIRE((xs1[k] >= xs2[k]) == res[6 * k + 5]);
    }

    CUDA_REQUIRE(cudaFree(d_xs1));
    CUDA_REQUIRE(cudaFree(d_xs2));
    CUDA_REQUIRE(cudaFree(d_res));
}
TEST_CASE("cuda bigint comparison", "[bigint][cuda]") {
    std::random_device rd;
    test_cuda_bigint_comparison<Int160>(rd());
    test_cuda_bigint_comparison<Int160_2>(rd());
}

template <typename Int>
__global__ static void test_cuda_bigint_bit_ops_kernel(Int *xs) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = 4 * id;
    Int res, tmp1, tmp2;
    int offset = 0;

    tmp1 = xs[idx + offset];
    tmp2 = xs[idx + offset + 1];
    res.bit_and(tmp1, tmp2);
    xs[idx + offset] = res;
    ++offset;

    tmp1 = xs[idx + offset];
    tmp2 = xs[idx + offset + 1];
    res.bit_or(tmp1, tmp2);
    xs[idx + offset] = res;
    ++offset;

    tmp1 = xs[idx + offset];
    tmp2 = xs[idx + offset + 1];
    res.bit_xor(tmp1, tmp2);
    xs[idx + offset] = res;
    ++offset;

    tmp1 = xs[idx + offset];
    res.bit_not(tmp1);
    xs[idx + offset] = res;
    ++offset;
}
template <typename Int>
static void test_cuda_bigint_bit_ops(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    constexpr int BN = 6;
    constexpr int TN = 64;
    constexpr int N = BN * TN;
    constexpr int K = 4;
    constexpr int M = K * N;
    constexpr size_t bytes = sizeof(Int) * M;

    Int xs[M], r_xs[M], tmp;
    for (size_t k = 0; k < M; ++k) {
        Int::sample(xs[k], rng);
    }

    Int *d_xs;

    CUDA_REQUIRE(cudaMalloc(&d_xs, bytes));
    CUDA_REQUIRE(cudaMemcpyAsync(d_xs, xs, bytes, cudaMemcpyHostToDevice));
    test_cuda_bigint_bit_ops_kernel<Int><<<BN, TN>>>(d_xs);
    CUDA_REQUIRE(cudaMemcpyAsync(r_xs, d_xs, bytes, cudaMemcpyDeviceToHost));

    for (size_t k = 0; k < N; ++k) {
        int offset = 0;
        Int *x = xs + K * k;
        tmp.bit_and(x[offset + 0], x[offset + 1]);
        x[offset + 0] = tmp;
        ++offset;
        tmp.bit_or(x[offset + 0], x[offset + 1]);
        x[offset + 0] = tmp;
        ++offset;
        tmp.bit_xor(x[offset + 0], x[offset + 1]);
        x[offset + 0] = tmp;
        ++offset;
        tmp.bit_not(x[offset + 0]);
        x[offset + 0] = tmp;
        ++offset;
    }

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    for (int k = 0; k < M; ++k) {
        CAPTURE(k);
        REQUIRE(xs[k] == r_xs[k]);
    }

    CUDA_REQUIRE(cudaFree(d_xs));
}
TEST_CASE("cuda bigint bit operations", "[bigint][cuda]") {
    std::random_device rd;
    test_cuda_bigint_bit_ops<Int160>(rd());
    test_cuda_bigint_bit_ops<Int160_2>(rd());
}

template <typename Int>
__global__ static void test_cuda_bigint_shift_kernel(Int *xs) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    Int *x = xs + 12 * id;
    Int tmp;
    int offset = 0;

#define test_helper(method, bit_n)                                             \
    do {                                                                       \
        tmp = x[offset];                                                       \
        tmp.template method<(bit_n)>();                                        \
        x[offset] = tmp;                                                       \
        ++offset;                                                              \
    } while (0)

    test_helper(shift_left, 0);
    test_helper(shift_left, 3);
    test_helper(shift_left, 32);
    test_helper(shift_left, 33);
    test_helper(shift_left, 66);
    test_helper(shift_left, 32 * 5);
    test_helper(shift_right, 0);
    test_helper(shift_right, 3);
    test_helper(shift_right, 32);
    test_helper(shift_right, 33);
    test_helper(shift_right, 66);
    test_helper(shift_right, 32 * 5);

#undef test_helper
}
template <typename Int>
static void test_bigint_shift(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    constexpr int BN = 6;
    constexpr int TN = 64;
    constexpr int N = BN * TN;
    constexpr int K = 12;
    constexpr int M = K * N;
    constexpr size_t bytes = sizeof(Int) * M;

    Int xs[M], r_xs[M], tmp;
    for (size_t k = 0; k < M; ++k) {
        Int::sample(xs[k], rng);
    }

    Int *d_xs;

    CUDA_REQUIRE(cudaMalloc(&d_xs, bytes));
    CUDA_REQUIRE(cudaMemcpyAsync(d_xs, xs, bytes, cudaMemcpyHostToDevice));
    test_cuda_bigint_shift_kernel<Int><<<BN, TN>>>(d_xs);
    CUDA_REQUIRE(cudaMemcpyAsync(r_xs, d_xs, bytes, cudaMemcpyDeviceToHost));

#define test_helper(method, bit_n)                                             \
    do {                                                                       \
        tmp = x[offset];                                                       \
        tmp.template method<(bit_n)>();                                        \
        x[offset] = tmp;                                                       \
        ++offset;                                                              \
    } while (0)

    for (size_t k = 0; k < N; ++k) {
        int offset = 0;
        Int *x = xs + K * k;

        test_helper(shift_left, 0);
        test_helper(shift_left, 3);
        test_helper(shift_left, 32);
        test_helper(shift_left, 33);
        test_helper(shift_left, 66);
        test_helper(shift_left, 32 * 5);
        test_helper(shift_right, 0);
        test_helper(shift_right, 3);
        test_helper(shift_right, 32);
        test_helper(shift_right, 33);
        test_helper(shift_right, 66);
        test_helper(shift_right, 32 * 5);
    }

#undef test_helper

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    for (int k = 0; k < M; ++k) {
        CAPTURE(k);
        REQUIRE(xs[k] == r_xs[k]);
    }

    CUDA_REQUIRE(cudaFree(d_xs));
}
TEST_CASE("cuda bigint shift", "[bigint][cuda]") {
    std::random_device rd;
    test_bigint_shift<Int160>(rd());
    test_bigint_shift<Int160_2>(rd());
}

template <typename Int>
__global__ static void test_add_kernel(Int *sum, Int *xs, Int *ys) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    Int x = xs[id], y = ys[id], s;
    Int::add(s, x, y);
    sum[id] = s;
}
template <typename Int>
static void test_add(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    Int x, y, sum, r_sum;
    Int::sample(x, rng);
    Int::sample(y, rng);
    Int::add(sum, x, y);

    Int *d_x, *d_y, *d_sum;
    CUDA_REQUIRE(cudaMalloc(&d_x, sizeof(Int)));
    CUDA_REQUIRE(cudaMalloc(&d_y, sizeof(Int)));
    CUDA_REQUIRE(cudaMalloc(&d_sum, sizeof(Int)));

    CUDA_REQUIRE(cudaMemcpyAsync(d_x, &x, sizeof(Int), cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_y, &y, sizeof(Int), cudaMemcpyHostToDevice));
    test_add_kernel<Int><<<1, 1>>>(d_sum, d_x, d_y);
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_sum, d_sum, sizeof(Int), cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(sum == r_sum);

    CUDA_REQUIRE(cudaFree(d_x));
    CUDA_REQUIRE(cudaFree(d_y));
    CUDA_REQUIRE(cudaFree(d_sum));
}
TEST_CASE("cuda bigint add", "[add_group][bigint][cuda]") {
    std::random_device rd;
    test_add<Int160>(rd());
    test_add<Int160_2>(rd());
}

template <typename Int>
__global__ static void test_sub_kernel(Int *diff, Int *xs, Int *ys) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    Int x = xs[id], y = ys[id], s;
    Int::sub(s, x, y);
    diff[id] = s;
}
template <typename Int>
static void test_sub(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    Int x, y, diff, r_diff;
    Int::sample(x, rng);
    Int::sample(y, rng);
    Int::sub(diff, x, y);

    Int *d_x, *d_y, *d_diff;
    CUDA_REQUIRE(cudaMalloc(&d_x, sizeof(Int)));
    CUDA_REQUIRE(cudaMalloc(&d_y, sizeof(Int)));
    CUDA_REQUIRE(cudaMalloc(&d_diff, sizeof(Int)));

    CUDA_REQUIRE(cudaMemcpyAsync(d_x, &x, sizeof(Int), cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_y, &y, sizeof(Int), cudaMemcpyHostToDevice));
    test_sub_kernel<Int><<<1, 1>>>(d_diff, d_x, d_y);
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_diff, d_diff, sizeof(Int), cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(diff == r_diff);

    CUDA_REQUIRE(cudaFree(d_x));
    CUDA_REQUIRE(cudaFree(d_y));
    CUDA_REQUIRE(cudaFree(d_diff));
}
TEST_CASE("cuda bigint sub", "[add_group][bigint][cuda]") {
    std::random_device rd;
    test_sub<Int160>(rd());
    test_sub<Int160_2>(rd());
}
