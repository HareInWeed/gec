#include "configured_catch.hpp"

#include <cinttypes>
#include <gec/curve/data/point.hpp>
#include <gec/utils/context.hpp>

template <typename T, T expr>
struct Eval {
    constexpr static T value = expr;
};

TEST_CASE("context", "[utils]") {
    using namespace gec::utils;

    Context<4 + 2 + 1 + 1 + 4, alignof(uint32_t), 0> ctx;

    auto &ctx_view =
        ctx.view_as<uint32_t, uint16_t, uint8_t, uint8_t, uint32_t>();

    ctx_view.get<0>() = 0x01010101u;
    ctx_view.get<1>() = 0x0202u;
    ctx_view.get<2>() = 0x03u;
    ctx_view.get<3>() = 0x04u;
    ctx_view.get<4>() = 0x05050505u;

    REQUIRE(0x01010101u == ctx_view.get<0>());
    REQUIRE(0x0202u == ctx_view.get<1>());
    REQUIRE(0x03u == ctx_view.get<2>());
    REQUIRE(0x04u == ctx_view.get<3>());
    REQUIRE(0x05050505u == ctx_view.get<4>());

    auto &ctx_view1 = ctx.view_as<uint32_t>()
                          .rest()
                          .view_as<uint16_t, uint8_t, uint8_t, uint32_t>();
    REQUIRE(0x0202u == ctx_view1.get<0>());
    REQUIRE(0x03u == ctx_view1.get<1>());
    REQUIRE(0x04u == ctx_view1.get<2>());
    REQUIRE(0x05050505u == ctx_view1.get<3>());

    auto &ctx_view2 = ctx_view1.view_as<uint16_t>()
                          .rest()
                          .view_as<uint8_t, uint8_t, uint32_t>();
    REQUIRE(0x03u == ctx_view2.get<0>());
    REQUIRE(0x04u == ctx_view2.get<1>());
    REQUIRE(0x05050505u == ctx_view2.get<2>());

    auto &ctx_view3 =
        ctx_view2.view_as<uint8_t>().rest().view_as<uint8_t, uint32_t>();
    REQUIRE(0x04u == ctx_view3.get<0>());
    REQUIRE(0x05050505u == ctx_view3.get<1>());

    auto &ctx_view4 = ctx_view3.view_as<uint8_t>().rest().view_as<uint32_t>();
    REQUIRE(0x05050505u == ctx_view4.get<0>());
}
