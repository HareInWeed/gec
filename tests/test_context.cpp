#include "configured_catch.hpp"

#include <gec/curve/data/point.hpp>
#include <gec/curve/data/point_context.hpp>
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

TEST_CASE("point context", "[curve]") {
    using namespace gec::utils;
    using namespace gec::curve;

    Point<int, 5> p;

    PointContext<Point<int, 3>, 4, 2> ctx;

    REQUIRE(10 == ctx.capacity);
    REQUIRE(2 == ctx.point_capacity);

    REQUIRE(0 == ctx.get<0>());
    REQUIRE(0 == ctx.get<1>());
    REQUIRE(0 == ctx.get<2>());
    REQUIRE(0 == ctx.get<3>());
    REQUIRE(0 == ctx.get<4>());
    REQUIRE(0 == ctx.get<5>());
    REQUIRE(0 == ctx.get<6>());
    REQUIRE(0 == ctx.get<7>());
    REQUIRE(0 == ctx.get<8>());
    REQUIRE(0 == ctx.get<9>());

    ctx.get<0>() = 0;
    ctx.get<1>() = 1;
    ctx.get<2>() = 2;
    ctx.get<3>() = 3;
    ctx.get<4>() = 4;
    ctx.get<5>() = 5;
    ctx.get<6>() = 6;
    ctx.get<7>() = 7;
    ctx.get<8>() = 8;
    ctx.get<9>() = 9;

    REQUIRE(0 == ctx.get<0>());
    REQUIRE(1 == ctx.get<1>());
    REQUIRE(2 == ctx.get<2>());
    REQUIRE(3 == ctx.get<3>());
    REQUIRE(4 == ctx.get<4>());
    REQUIRE(5 == ctx.get<5>());
    REQUIRE(6 == ctx.get<6>());
    REQUIRE(7 == ctx.get<7>());
    REQUIRE(8 == ctx.get<8>());
    REQUIRE(9 == ctx.get<9>());

    REQUIRE(4 == ctx.get_p<1>().get<0>());
    REQUIRE(5 == ctx.get_p<1>().get<1>());
    REQUIRE(6 == ctx.get_p<1>().get<2>());
    REQUIRE(7 == ctx.get_p<0>().get<0>());
    REQUIRE(8 == ctx.get_p<0>().get<1>());
    REQUIRE(9 == ctx.get_p<0>().get<2>());

    auto &ctx_view1 = ctx.rest<2>();

    REQUIRE(8 == ctx_view1.capacity);
    REQUIRE(2 == ctx_view1.point_capacity);

    REQUIRE(2 == ctx_view1.get<0>());
    REQUIRE(3 == ctx_view1.get<1>());
    REQUIRE(4 == ctx_view1.get<2>());
    REQUIRE(5 == ctx_view1.get<3>());
    REQUIRE(6 == ctx_view1.get<4>());
    REQUIRE(7 == ctx_view1.get<5>());
    REQUIRE(8 == ctx_view1.get<6>());
    REQUIRE(9 == ctx_view1.get<7>());

    REQUIRE(4 == ctx_view1.get_p<1>().get<0>());
    REQUIRE(5 == ctx_view1.get_p<1>().get<1>());
    REQUIRE(6 == ctx_view1.get_p<1>().get<2>());
    REQUIRE(7 == ctx_view1.get_p<0>().get<0>());
    REQUIRE(8 == ctx_view1.get_p<0>().get<1>());
    REQUIRE(9 == ctx_view1.get_p<0>().get<2>());

    auto &ctx_view2 = ctx_view1.rest<3>();

    REQUIRE(5 == ctx_view2.capacity);
    REQUIRE(1 == ctx_view2.point_capacity);

    REQUIRE(5 == ctx_view2.get<0>());
    REQUIRE(6 == ctx_view2.get<1>());
    REQUIRE(7 == ctx_view2.get<2>());
    REQUIRE(8 == ctx_view2.get<3>());
    REQUIRE(9 == ctx_view2.get<4>());

    REQUIRE(7 == ctx_view2.get_p<0>().get<0>());
    REQUIRE(8 == ctx_view2.get_p<0>().get<1>());
    REQUIRE(9 == ctx_view2.get_p<0>().get<2>());

    auto &ctx_view3 = ctx_view2.rest<0, 1>();

    REQUIRE(2 == ctx_view3.capacity);
    REQUIRE(0 == ctx_view3.point_capacity);

    REQUIRE(5 == ctx_view3.get<0>());
    REQUIRE(6 == ctx_view3.get<1>());

    auto &ctx_view4 = ctx.rest<5, 1>();

    REQUIRE(2 == ctx_view4.capacity);
    REQUIRE(0 == ctx_view4.point_capacity);

    REQUIRE(5 == ctx_view4.get<0>());
    REQUIRE(6 == ctx_view4.get<1>());
}
