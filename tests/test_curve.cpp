#include "common.hpp"

#include <gec/bigint.hpp>
#include <gec/curve.hpp>

#include "curve.hpp"

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;
using namespace curve;

TEST_CASE("point2", "[curve]") {
    using F = Field160;
    Point<F, 2> p(F(0x0u, 0x0u, 0x0u, 0x0u, 0x1u),
                  F(0x1u, 0x0u, 0x0u, 0x0u, 0x0u));
    REQUIRE(p.x().array()[0] == 1);
    REQUIRE(p.y().array()[F::LimbN - 1] == 1);
}

TEST_CASE("affine", "[curve][affine]") {
    using C = CurveA;
    using F = Field160;
    const auto &RR = reinterpret_cast<const F &>(RR_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           RR);
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           RR);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           RR);
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           RR);
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2, ctx));

    C sum;
    C::add(sum, p1, p2, ctx);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           RR);
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           RR);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CHECK(C::eq(expected, sum));
}

TEST_CASE("affine bench", "[curve][affine][bench]") {
    using C = CurveA2;
    using F = Field160_2;

    const auto &RR = reinterpret_cast<const F &>(RR2_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u}, RR);
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu}, RR);

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu}, RR);
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u}, RR);

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2, ctx);
        return &sum;
    };
}

TEST_CASE("projective", "[curve][projective]") {
    using C = CurveP;
    using F = Field160;
    const auto &RR = reinterpret_cast<const F &>(RR_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           RR);
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           RR);
    C::from_affine(p1);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           RR);
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           RR);
    C::from_affine(p2);
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2, ctx));

    C sum;
    C::add(sum, p1, p2, ctx);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           RR);
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           RR);
    C::from_affine(expected);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CHECK(C::eq(expected, sum, ctx));
}

TEST_CASE("projective bench", "[curve][projective][bench]") {
    using C = CurveP2;
    using F = Field160_2;

    const auto &RR = reinterpret_cast<const F &>(RR2_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u}, RR);
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu}, RR);
    C::from_affine(p1);

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu}, RR);
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u}, RR);
    C::from_affine(p2);

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2, ctx);
        return &sum;
    };
}

TEST_CASE("jacobian", "[curve][jacobian]") {
    using C = CurveJ;
    using F = Field160;
    const auto &RR = reinterpret_cast<const F &>(RR_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           RR);
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           RR);
    C::from_affine(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           RR);
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           RR);
    C::from_affine(p2);
    REQUIRE(C::on_curve(p2, ctx));

    C sum;
    C::add(sum, p1, p2, ctx);
    REQUIRE(C::on_curve(sum, ctx));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           RR);
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           RR);
    C::from_affine(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CAPTURE(expected, sum);
    CHECK(C::eq(expected, sum, ctx));
}

TEST_CASE("jacobian bench", "[curve][jacobian][bench]") {
    using C = CurveJ2;
    using F = Field160_2;

    const auto &RR = reinterpret_cast<const F &>(RR2_160);
    const F One(1);

    F::Context ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u}, RR);
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu}, RR);
    C::from_affine(p1);

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu}, RR);
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u}, RR);
    C::from_affine(p2);

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2, ctx);
        return &sum;
    };
}