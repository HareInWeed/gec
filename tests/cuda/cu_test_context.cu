#include <configured_catch.hpp>

#include <cinttypes>
#include <gec/curve/data/point.hpp>
#include <gec/utils/context.hpp>

#include "cuda_common.cuh"

using namespace gec::utils;

#define ALIGNED(x)                                                             \
    reinterpret_cast<uintptr_t>(&(x)) % alignof(decltype((x))) == 0

using CTX = Context<4 + 2 + 1 + 1 + 4, alignof(uint32_t), 0>;

__global__ static void
test_kernel1(uint32_t *GEC_RSTRCT c0, bool *GEC_RSTRCT a0,
             uint16_t *GEC_RSTRCT c1, bool *GEC_RSTRCT a1,
             uint8_t *GEC_RSTRCT c2, bool *GEC_RSTRCT a2,
             uint8_t *GEC_RSTRCT c3, bool *GEC_RSTRCT a3,
             uint32_t *GEC_RSTRCT c4, bool *GEC_RSTRCT a4) {
    CTX ctx;
    auto &ctx_view =
        ctx.view_as<uint32_t, uint16_t, uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c0;
    ctx_view.get<1>() = *c1;
    ctx_view.get<2>() = *c2;
    ctx_view.get<3>() = *c3;
    ctx_view.get<4>() = *c4;
    *c0 = ctx_view.get<0>();
    *c1 = ctx_view.get<1>();
    *c2 = ctx_view.get<2>();
    *c3 = ctx_view.get<3>();
    *c4 = ctx_view.get<4>();
    *a0 = ALIGNED(ctx_view.get<0>());
    *a1 = ALIGNED(ctx_view.get<1>());
    *a2 = ALIGNED(ctx_view.get<2>());
    *a3 = ALIGNED(ctx_view.get<3>());
    *a4 = ALIGNED(ctx_view.get<4>());
}
__global__ static void test_kernel1(uint32_t *GEC_RSTRCT c0,
                                    uint16_t *GEC_RSTRCT c1,
                                    uint8_t *GEC_RSTRCT c2,
                                    uint8_t *GEC_RSTRCT c3,
                                    uint32_t *GEC_RSTRCT c4) {
    CTX ctx;
    auto &ctx_view =
        ctx.view_as<uint32_t, uint16_t, uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c0;
    ctx_view.get<1>() = *c1;
    ctx_view.get<2>() = *c2;
    ctx_view.get<3>() = *c3;
    ctx_view.get<4>() = *c4;
    *c0 = ctx_view.get<0>();
    *c1 = ctx_view.get<1>();
    *c2 = ctx_view.get<2>();
    *c3 = ctx_view.get<3>();
    *c4 = ctx_view.get<4>();
}

__global__ static void
test_kernel2(uint32_t *GEC_RSTRCT, bool *GEC_RSTRCT, uint16_t *GEC_RSTRCT c1,
             bool *GEC_RSTRCT a1, uint8_t *GEC_RSTRCT c2, bool *GEC_RSTRCT a2,
             uint8_t *GEC_RSTRCT c3, bool *GEC_RSTRCT a3,
             uint32_t *GEC_RSTRCT c4, bool *GEC_RSTRCT a4) {
    CTX ctx;

    auto &ctx_view = ctx.view_as<uint32_t>()
                         .rest()
                         .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c1;
    ctx_view.get<1>() = *c2;
    ctx_view.get<2>() = *c3;
    ctx_view.get<3>() = *c4;
    *c1 = ctx_view.get<0>();
    *c2 = ctx_view.get<1>();
    *c3 = ctx_view.get<2>();
    *c4 = ctx_view.get<3>();
    *a1 = ALIGNED(ctx_view.get<0>());
    *a2 = ALIGNED(ctx_view.get<1>());
    *a3 = ALIGNED(ctx_view.get<2>());
    *a4 = ALIGNED(ctx_view.get<3>());
}
__global__ static void test_kernel2(uint32_t *GEC_RSTRCT,
                                    uint16_t *GEC_RSTRCT c1,
                                    uint8_t *GEC_RSTRCT c2,
                                    uint8_t *GEC_RSTRCT c3,
                                    uint32_t *GEC_RSTRCT c4) {
    CTX ctx;

    auto &ctx_view = ctx.view_as<uint32_t>()
                         .rest()
                         .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c1;
    ctx_view.get<1>() = *c2;
    ctx_view.get<2>() = *c3;
    ctx_view.get<3>() = *c4;
    *c1 = ctx_view.get<0>();
    *c2 = ctx_view.get<1>();
    *c3 = ctx_view.get<2>();
    *c4 = ctx_view.get<3>();
}

__global__ static void test_kernel3(uint32_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint16_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT c2, bool *GEC_RSTRCT a2,
                                    uint8_t *GEC_RSTRCT c3, bool *GEC_RSTRCT a3,
                                    uint32_t *GEC_RSTRCT c4,
                                    bool *GEC_RSTRCT a4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view = ctx_view1.view_as<uint16_t>()
                         .rest()
                         .view_as<uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c2;
    ctx_view.get<1>() = *c3;
    ctx_view.get<2>() = *c4;
    *c2 = ctx_view.get<0>();
    *c3 = ctx_view.get<1>();
    *c4 = ctx_view.get<2>();
    *a2 = ALIGNED(ctx_view.get<0>());
    *a3 = ALIGNED(ctx_view.get<1>());
    *a4 = ALIGNED(ctx_view.get<2>());
}
__global__ static void test_kernel3(uint32_t *GEC_RSTRCT, uint16_t *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT c2,
                                    uint8_t *GEC_RSTRCT c3,
                                    uint32_t *GEC_RSTRCT c4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view = ctx_view1.view_as<uint16_t>()
                         .rest()
                         .view_as<uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = *c2;
    ctx_view.get<1>() = *c3;
    ctx_view.get<2>() = *c4;
    *c2 = ctx_view.get<0>();
    *c3 = ctx_view.get<1>();
    *c4 = ctx_view.get<2>();
}

__global__ static void test_kernel4(uint32_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint16_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT c3, bool *GEC_RSTRCT a3,
                                    uint32_t *GEC_RSTRCT c4,
                                    bool *GEC_RSTRCT a4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view2 = ctx_view1.view_as<uint16_t>()
                          .rest()
                          .view_as<uint8_t, uint8_t, uint32_t>();
    auto &ctx_view =
        ctx_view2.view_as<uint8_t>().rest().view_as<uint8_t, uint32_t>();

    ctx_view.get<0>() = *c3;
    ctx_view.get<1>() = *c4;
    *c3 = ctx_view.get<0>();
    *c4 = ctx_view.get<1>();
    *a3 = ALIGNED(ctx_view.get<0>());
    *a4 = ALIGNED(ctx_view.get<1>());
}
__global__ static void test_kernel4(uint32_t *GEC_RSTRCT, uint16_t *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT, uint8_t *GEC_RSTRCT c3,
                                    uint32_t *GEC_RSTRCT c4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view2 = ctx_view1.view_as<uint16_t>()
                          .rest()
                          .view_as<uint8_t, uint8_t, uint32_t>();
    auto &ctx_view =
        ctx_view2.view_as<uint8_t>().rest().view_as<uint8_t, uint32_t>();

    ctx_view.get<0>() = *c3;
    ctx_view.get<1>() = *c4;
    *c3 = ctx_view.get<0>();
    *c4 = ctx_view.get<1>();
}

__global__ static void test_kernel5(uint32_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint16_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT, bool *GEC_RSTRCT,
                                    uint32_t *GEC_RSTRCT c4,
                                    bool *GEC_RSTRCT a4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view2 = ctx_view1.view_as<uint16_t>()
                          .rest()
                          .view_as<uint8_t, uint8_t, uint32_t>();
    auto &ctx_view3 =
        ctx_view2.view_as<uint8_t>().rest().view_as<uint8_t, uint32_t>();
    auto &ctx_view = ctx_view3.view_as<uint8_t>().rest().view_as<uint32_t>();

    ctx_view.get<0>() = *c4;
    *c4 = ctx_view.get<0>();
    *a4 = ALIGNED(ctx_view.get<0>());
}
__global__ static void test_kernel5(uint32_t *GEC_RSTRCT, uint16_t *GEC_RSTRCT,
                                    uint8_t *GEC_RSTRCT, uint8_t *GEC_RSTRCT,
                                    uint32_t *GEC_RSTRCT c4) {
    CTX ctx;

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    auto &ctx_view2 = ctx_view1.view_as<uint16_t>()
                          .rest()
                          .view_as<uint8_t, uint8_t, uint32_t>();
    auto &ctx_view3 =
        ctx_view2.view_as<uint8_t>().rest().view_as<uint8_t, uint32_t>();
    auto &ctx_view = ctx_view3.view_as<uint8_t>().rest().view_as<uint32_t>();

    ctx_view.get<0>() = *c4;
    *c4 = ctx_view.get<0>();
}

TEST_CASE("cuda context", "[cuda][utils]") {
    uint32_t c0 = 0x01010101u, *d_c0, r0;
    uint16_t c1 = 0x0202u, *d_c1, r1;
    uint8_t c2 = 0x03u, *d_c2, r2;
    uint8_t c3 = 0x04u, *d_c3, r3;
    uint32_t c4 = 0x05050505u, *d_c4, r4;
    bool a0, a1, a2, a3, a4;
    bool *d_a0, *d_a1, *d_a2, *d_a3, *d_a4;

#define GEC_H2D                                                                \
    do {                                                                       \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(d_c0, &c0, sizeof(c0), cudaMemcpyHostToDevice));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(d_c1, &c1, sizeof(c1), cudaMemcpyHostToDevice));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(d_c2, &c2, sizeof(c2), cudaMemcpyHostToDevice));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(d_c3, &c3, sizeof(c3), cudaMemcpyHostToDevice));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(d_c4, &c4, sizeof(c4), cudaMemcpyHostToDevice));        \
    } while (0)

#define GEC_D2H                                                                \
    do {                                                                       \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&r0, d_c0, sizeof(r0), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&r1, d_c1, sizeof(r1), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&r2, d_c2, sizeof(r2), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&r3, d_c3, sizeof(r3), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&r4, d_c4, sizeof(r4), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&a0, d_a0, sizeof(a0), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&a1, d_a1, sizeof(a1), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&a2, d_a2, sizeof(a2), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&a3, d_a3, sizeof(a3), cudaMemcpyDeviceToHost));        \
        CUDA_REQUIRE(                                                          \
            cudaMemcpy(&a4, d_a4, sizeof(a4), cudaMemcpyDeviceToHost));        \
    } while (0)

    CUDA_REQUIRE(cudaMalloc(&d_c0, sizeof(*d_c0)));
    CUDA_REQUIRE(cudaMalloc(&d_c1, sizeof(*d_c1)));
    CUDA_REQUIRE(cudaMalloc(&d_c2, sizeof(*d_c2)));
    CUDA_REQUIRE(cudaMalloc(&d_c3, sizeof(*d_c3)));
    CUDA_REQUIRE(cudaMalloc(&d_c4, sizeof(*d_c4)));
    CUDA_REQUIRE(cudaMalloc(&d_a0, sizeof(*d_a0)));
    CUDA_REQUIRE(cudaMalloc(&d_a1, sizeof(*d_a1)));
    CUDA_REQUIRE(cudaMalloc(&d_a2, sizeof(*d_a2)));
    CUDA_REQUIRE(cudaMalloc(&d_a3, sizeof(*d_a3)));
    CUDA_REQUIRE(cudaMalloc(&d_a4, sizeof(*d_a4)));

    GEC_H2D;
    test_kernel1<<<1, 1>>>(d_c0, d_a0, d_c1, d_a1, d_c2, d_a2, d_c3, d_a3, d_c4,
                           d_a4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c0 == r0);
    REQUIRE(c1 == r1);
    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);
    REQUIRE(a0);
    REQUIRE(a1);
    REQUIRE(a2);
    REQUIRE(a3);
    REQUIRE(a4);

    GEC_H2D;
    test_kernel1<<<1, 1>>>(d_c0, d_c1, d_c2, d_c3, d_c4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c0 == r0);
    REQUIRE(c1 == r1);
    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);

    GEC_H2D;
    test_kernel2<<<1, 1>>>(d_c0, d_a0, d_c1, d_a1, d_c2, d_a2, d_c3, d_a3, d_c4,
                           d_a4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c1 == r1);
    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);
    REQUIRE(a1);
    REQUIRE(a2);
    REQUIRE(a3);
    REQUIRE(a4);

    GEC_H2D;
    test_kernel2<<<1, 1>>>(d_c0, d_c1, d_c2, d_c3, d_c4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c1 == r1);
    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);

    GEC_H2D;
    test_kernel3<<<1, 1>>>(d_c0, d_a0, d_c1, d_a1, d_c2, d_a2, d_c3, d_a3, d_c4,
                           d_a4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);
    REQUIRE(a2);
    REQUIRE(a3);
    REQUIRE(a4);

    GEC_H2D;
    test_kernel3<<<1, 1>>>(d_c0, d_c1, d_c2, d_c3, d_c4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c2 == r2);
    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);

    GEC_H2D;
    test_kernel4<<<1, 1>>>(d_c0, d_a0, d_c1, d_a1, d_c2, d_a2, d_c3, d_a3, d_c4,
                           d_a4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);
    REQUIRE(a3);
    REQUIRE(a4);

    GEC_H2D;
    test_kernel4<<<1, 1>>>(d_c0, d_c1, d_c2, d_c3, d_c4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c3 == r3);
    REQUIRE(c4 == r4);

    GEC_H2D;
    test_kernel5<<<1, 1>>>(d_c0, d_a0, d_c1, d_a1, d_c2, d_a2, d_c3, d_a3, d_c4,
                           d_a4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c4 == r4);
    REQUIRE(a4);

    GEC_H2D;
    test_kernel5<<<1, 1>>>(d_c0, d_c1, d_c2, d_c3, d_c4);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    GEC_D2H;

    REQUIRE(c4 == r4);

    cudaFree(d_c0);
    cudaFree(d_c1);
    cudaFree(d_c2);
    cudaFree(d_c3);
    cudaFree(d_c4);
    cudaFree(d_a0);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_a3);
    cudaFree(d_a4);

#undef GEC_H2D
#undef GEC_D2H
}
