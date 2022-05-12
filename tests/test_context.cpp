#include "configured_catch.hpp"

#include <gec/utils/context.hpp>

TEST_CASE("context", "[utils]") {
    using namespace gec::utils;

    Context<int, 4> ctx;

    ctx.get<0>() = 1;
    ctx.get<1>() = 2;
    ctx.rest<1>().rest<1>().get<0>() = 3;
    ctx.rest<2>().get<1>() = 4;

    REQUIRE(1 == ctx.get<0>());
    REQUIRE(2 == ctx.get<1>());
    REQUIRE(3 == ctx.get<2>());
    REQUIRE(4 == ctx.get<3>());

    REQUIRE(2 == ctx.rest<1>().get<0>());
    REQUIRE(3 == ctx.rest<1>().get<1>());
    REQUIRE(4 == ctx.rest<1>().get<2>());

    REQUIRE(3 == ctx.rest<2>().get<0>());
    REQUIRE(4 == ctx.rest<2>().get<1>());

    REQUIRE(3 == ctx.rest<1>().rest<1>().get<0>());
    REQUIRE(4 == ctx.rest<1>().rest<1>().get<1>());

    REQUIRE(4 == ctx.rest<3>().get<0>());
    REQUIRE(4 == ctx.rest<1>().rest<2>().get<0>());
    REQUIRE(4 == ctx.rest<2>().rest<1>().get<0>());
    REQUIRE(4 == ctx.rest<1>().rest<1>().rest<1>().get<0>());

    Context<int, 4> ctx1(5, 6, 7, 8);
    REQUIRE(5 == ctx1.get<0>());
    REQUIRE(6 == ctx1.get<1>());
    REQUIRE(7 == ctx1.get<2>());
    REQUIRE(8 == ctx1.get<3>());
    --ctx1.get<0>();
    --ctx1.rest<1>().get<0>();
    --ctx1.rest<2>().get<0>();
    --ctx1.rest<3>().get<0>();
    REQUIRE(4 == ctx1.get<0>());
    REQUIRE(5 == ctx1.get<1>());
    REQUIRE(6 == ctx1.get<2>());
    REQUIRE(7 == ctx1.get<3>());

    int a = 9, b = 10, c = 11, d = 12;
    Context<int &, 4> ctx2(a, b, c, d);
    REQUIRE(9 == ctx2.get<0>());
    REQUIRE(10 == ctx2.get<1>());
    REQUIRE(11 == ctx2.get<2>());
    REQUIRE(12 == ctx2.get<3>());
    ctx2.get<0>() += 2;
    ctx2.rest<1>().get<0>() += 2;
    ctx2.rest<2>().get<0>() += 2;
    ctx2.rest<3>().get<0>() += 2;
    REQUIRE(11 == ctx2.get<0>());
    REQUIRE(12 == ctx2.get<1>());
    REQUIRE(13 == ctx2.get<2>());
    REQUIRE(14 == ctx2.get<3>());
    REQUIRE(11 == a);
    REQUIRE(12 == b);
    REQUIRE(13 == c);
    REQUIRE(14 == d);
}
