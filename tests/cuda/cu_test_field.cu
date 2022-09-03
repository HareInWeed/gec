#define GEC_DEBUG

#include <gec/utils/cuda_utils.cuh>
#include <gec/utils/hash.hpp>

#include <common.hpp>

#include <field.hpp>

#include "cuda_common.cuh"

#include <configured_catch.hpp>

#include <thrust/random.h>

using namespace gec;
using namespace utils;

__global__ static void test_set_get_cc_cf_(bool *flags) {
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

__global__ static void test_add_cc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    add_cc_(vals[i], (a), (b));                                                \
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
    T vals[N];
    T *d_vals;
    size_t vals_size = sizeof(T) * N;

    bool *d_carries;
    size_t carries_size = sizeof(bool) * N;
    bool carries[N];

    CUDA_REQUIRE(cudaMalloc(&d_vals, vals_size));
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

    CUDA_REQUIRE(cudaFree(d_vals));
    CUDA_REQUIRE(cudaFree(d_carries));
}

__global__ static void test_addc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    addc_(vals[i], (a), (b));                                                  \
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

    test_addc_<<<40, 64>>>(d_vals, d_carries);
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

__global__ static void test_addc_cc_(uint32_t *vals, bool *carries) {
    int i = 0;

#define test_helper(carry, a, b)                                               \
    set_cc_cf_((carry));                                                       \
    addc_cc_(vals[i], (a), (b));                                               \
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

using SmallArray = gec::bigint::ArrayBE<LIMB_T, 3>;
GEC_DEF(SmallMod, static const SmallArray, 0x0, 0xb, 0x7);

template <typename Rng>
struct test_rng_init;
template <>
struct test_rng_init<thrust::random::ranlux24> {
    __device__ static GecRng<thrust::random::ranlux24> call(size_t seed,
                                                            size_t id) {
        gec::hash::hash_combine(seed, id);
        return make_gec_rng(thrust::random::ranlux24(seed));
    }
};
template <>
struct test_rng_init<curandStateXORWOW_t> {
    __device__ static GecRng<curandStateXORWOW_t> call(size_t seed, size_t id) {
        auto rng = make_gec_rng(curandStateXORWOW_t());
        curand_init(seed, id, 0, &rng.get_rng());
        return rng;
    }
};

template <typename Int, typename Rng>
__global__ static void test_sampling_kernel(size_t seed, Int *x) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;

    auto rng = test_rng_init<Rng>::call(seed, id);

    typename Int::template Context<> ctx;

    Int::sample(x[0], rng);

    Int::sample_non_zero(x[1], rng);

    Int::sample(x[2], x[1], rng);

    Int::sample(x[3], x[2], x[1], rng, ctx);

    Int::sample_inclusive(x[4], x[1], rng);

    Int::sample_inclusive(x[5], x[2], x[1], rng, ctx);
}

template <typename Int, typename Rng>
static void test_sampling(size_t seed) {
    Int *x, *d_x;
    size_t x_size = 6 * sizeof(Int);

    CUDA_REQUIRE(cudaMallocHost(&x, x_size));
    CUDA_REQUIRE(cudaMalloc(&d_x, x_size));

    test_sampling_kernel<Int, Rng><<<1, 1>>>(seed, d_x);
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
}

TEST_CASE("cuda random sampling", "[add_group][field][random][cuda]") {
    using F1 = Field160;
    using F2 = Field160_2;
    using G = GEC_BASE_ADD_GROUP(decltype(SmallMod), SmallMod);

    std::random_device rd;

    std::random_device::result_type seed;
    seed = rd();
    CAPTURE(seed);
    test_sampling<F1, curandStateXORWOW_t>(seed);
    seed = rd();
    CAPTURE(seed);
    test_sampling<F2, thrust::random::ranlux24>(seed);
    seed = rd();
    CAPTURE(seed);
    test_sampling<G, curandStateXORWOW_t>(seed);
}

template <typename Int>
__global__ static void test_neg_kernel(Int *neg_xs, Int *xs) {
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    Int neg_x, x = xs[id];
    Int::neg(neg_x, x);
    neg_xs[id] = neg_x;
}
template <typename Int>
static void test_neg(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    Int x, neg_x, ep_neg_x;
    Int::sample(x, rng);
    Int::neg(ep_neg_x, x);

    Int *d_neg_x, *d_x;
    CUDA_REQUIRE(cudaMalloc(&d_neg_x, sizeof(Int)));
    CUDA_REQUIRE(cudaMalloc(&d_x, sizeof(Int)));

    CUDA_REQUIRE(cudaMemcpyAsync(d_x, &x, sizeof(Int), cudaMemcpyHostToDevice));
    test_neg_kernel<Int><<<1, 1>>>(d_neg_x, d_x);
    CUDA_REQUIRE(
        cudaMemcpyAsync(&neg_x, d_neg_x, sizeof(Int), cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    REQUIRE(ep_neg_x == neg_x);

    CUDA_REQUIRE(cudaFree(d_x));
    CUDA_REQUIRE(cudaFree(d_neg_x));
}

TEST_CASE("cuda add group neg", "[add_group][field][cuda]") {
    std::random_device rd;
    test_neg<Field160>(rd());
    test_neg<Field160_2>(rd());
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
TEST_CASE("cuda add group add", "[add_group][field][cuda]") {
    std::random_device rd;
    test_add<Field160>(rd());
    test_add<Field160_2>(rd());
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
TEST_CASE("cuda add group sub", "[add_group][field][cuda]") {
    std::random_device rd;
    test_sub<Field160>(rd());
    test_sub<Field160_2>(rd());
}

template <typename Int>
__global__ static void test_mul_pow2_kernel(Int *pow) {
    Int::template mul_pow2<1>(pow[0]);
    Int::template mul_pow2<2>(pow[1]);
    Int::template mul_pow2<3>(pow[2]);
    Int::template mul_pow2<32>(pow[3]);
}
template <typename Int>
static void test_mul_pow2(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    constexpr int N = 4;
    constexpr size_t bytes = sizeof(Int) * N;

    Int pow[N], r_pow[N];

    for (int k = 0; k < N; ++k) {
        Int::sample(pow[k], rng);
        r_pow[k] = pow[k];
    }

    Int *d_pow;

    CUDA_REQUIRE(cudaMalloc(&d_pow, bytes));

    CUDA_REQUIRE(cudaMemcpy(d_pow, pow, bytes, cudaMemcpyHostToDevice));
    test_mul_pow2_kernel<Int><<<1, 1>>>(d_pow);
    CUDA_REQUIRE(cudaMemcpyAsync(r_pow, d_pow, bytes, cudaMemcpyDeviceToHost));

    Int::template mul_pow2<1>(pow[0]);
    Int::template mul_pow2<2>(pow[1]);
    Int::template mul_pow2<3>(pow[2]);
    Int::template mul_pow2<32>(pow[3]);

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    for (int k = 0; k < N; ++k) {
        CAPTURE(k);
        REQUIRE(pow[k] == r_pow[k]);
    }

    CUDA_REQUIRE(cudaFree(d_pow));
}
TEST_CASE("cuda mul_pow2", "[add_group][field][cuda]") {
    std::random_device rd;
    test_mul_pow2<Field160>(rd());
    test_mul_pow2<Field160_2>(rd());
}

template <typename Int>
__global__ static void test_montgomery_kernel(Int *d_x, Int *d_y, Int *d_mon_x,
                                              Int *d_mon_y, Int *d_mon_prod,
                                              Int *d_prod) {
    Int x = *d_x, y = *d_y, mon_x, mon_y, mon_prod, prod;
    Int::to_montgomery(mon_x, x);
    Int::to_montgomery(mon_y, y);
    Int::mul(mon_prod, mon_x, mon_y);
    Int::from_montgomery(prod, mon_prod);
    *d_mon_x = mon_x;
    *d_mon_y = mon_y;
    *d_mon_prod = mon_prod;
    *d_prod = prod;
}
template <typename Int>
static void test_montgomery(std::random_device::result_type seed) {
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    constexpr size_t bytes = sizeof(Int);

    Int x, y, r_mon_x, r_mon_y, r_mon_prod, r_prod;
    Int::sample(x, rng);
    Int::sample(y, rng);

    Int *d_x, *d_y, *d_mon_x, *d_mon_y, *d_mon_prod, *d_prod;

    CUDA_REQUIRE(cudaMalloc(&d_x, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_y, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_mon_x, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_mon_y, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_mon_prod, bytes));
    CUDA_REQUIRE(cudaMalloc(&d_prod, bytes));

    CUDA_REQUIRE(cudaMemcpyAsync(d_x, &x, bytes, cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpyAsync(d_y, &y, bytes, cudaMemcpyHostToDevice));
    test_montgomery_kernel<Int>
        <<<1, 1>>>(d_x, d_y, d_mon_x, d_mon_y, d_mon_prod, d_prod);
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_mon_x, d_mon_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_mon_y, d_mon_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(cudaMemcpyAsync(&r_mon_prod, d_mon_prod, bytes,
                                 cudaMemcpyDeviceToHost));
    CUDA_REQUIRE(
        cudaMemcpyAsync(&r_prod, d_prod, bytes, cudaMemcpyDeviceToHost));

    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());

    Int mon_x, mon_y, mon_prod, prod;
    printf("host: \n");
    Int::to_montgomery(mon_x, x);
    Int::to_montgomery(mon_y, y);
    Int::mul(mon_prod, mon_x, mon_y);
    Int::from_montgomery(prod, mon_prod);

    REQUIRE(mon_x == r_mon_x);
    REQUIRE(mon_y == r_mon_y);
    REQUIRE(mon_prod == r_mon_prod);
    REQUIRE(prod == r_prod);

    CUDA_REQUIRE(cudaFree(d_x));
    CUDA_REQUIRE(cudaFree(d_y));
    CUDA_REQUIRE(cudaFree(d_mon_x));
    CUDA_REQUIRE(cudaFree(d_mon_y));
    CUDA_REQUIRE(cudaFree(d_mon_prod));
    CUDA_REQUIRE(cudaFree(d_prod));
}
TEST_CASE("cuda montgomery mul", "[add_group][field][cuda]") {
    std::random_device rd;
    test_montgomery<Field160>(rd());
    test_montgomery<Field160_2>(rd());
}
