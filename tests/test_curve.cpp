#include "common.hpp"

#include <gec/bigint.hpp>
#include <gec/curve.hpp>

#include "curve.hpp"

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;
using namespace curve;

TEST_CASE("point", "[curve]") {
    using F = Field160;
    Point<F, 2> p(F(0x0u, 0x0u, 0x0u, 0x0u, 0x1u),
                  F(0x1u, 0x0u, 0x0u, 0x0u, 0x0u));
    CAPTURE(p.x(), p.y());
    REQUIRE(p.x().array()[0] == 1);
    REQUIRE(p.y().array()[F::LimbN - 1] == 1);
}

TEST_CASE("affine", "[curve][affine]") {
    using C = CurveA;
    using F = Field160;

    C::Context<> ctx;

    C test(F(1), F(1));
    REQUIRE(!C::on_curve(test, ctx));

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           F::r_sqr());
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           F::r_sqr());
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           F::r_sqr());
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           F::r_sqr());
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2, ctx));

    C sum;
    C::add(sum, p1, p2, ctx);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           F::r_sqr());
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           F::r_sqr());
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CHECK(C::eq(expected, sum));
}

TEST_CASE("affine bench", "[curve][affine][bench]") {
    using C = CurveA2;
    using F = Field160_2;

    C::Context<> ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u},
           F::r_sqr());
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu},
           F::r_sqr());

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu},
           F::r_sqr());
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u},
           F::r_sqr());

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2, ctx);
        return &sum;
    };
}

TEST_CASE("projective", "[curve][projective]") {
    using C = CurveP;
    using F = Field160;

    C::Context<> ctx;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           F::r_sqr());
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           F::r_sqr());
    C::from_affine(p1);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           F::r_sqr());
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           F::r_sqr());
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
           F::r_sqr());
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           F::r_sqr());
    C::from_affine(expected);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CHECK(C::eq(expected, sum, ctx));
}

TEST_CASE("projective bench", "[curve][projective][bench]") {
    using C = CurveP2;
    using F = Field160_2;

    C::Context<> ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u},
           F::r_sqr());
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu},
           F::r_sqr());
    C::from_affine(p1);

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu},
           F::r_sqr());
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u},
           F::r_sqr());
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

    C::Context<> ctx;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           F::r_sqr());
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           F::r_sqr());
    C::from_affine(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           F::r_sqr());
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           F::r_sqr());
    C::from_affine(p2);
    REQUIRE(C::on_curve(p2, ctx));

    C sum;
    C::add(sum, p1, p2, ctx);
    REQUIRE(C::on_curve(sum, ctx));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           F::r_sqr());
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           F::r_sqr());
    C::from_affine(expected);
    REQUIRE(C::on_curve(expected, ctx));

    CAPTURE(expected, sum);
    CHECK(C::eq(expected, sum, ctx));
}

TEST_CASE("jacobian bench", "[curve][jacobian][bench]") {
    using C = CurveJ2;
    using F = Field160_2;

    C::Context<> ctx;

    C p1;
    F::mul(p1.x(), {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u},
           F::r_sqr());
    F::mul(p1.y(), {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu},
           F::r_sqr());
    C::from_affine(p1);

    C p2;
    F::mul(p2.x(), {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu},
           F::r_sqr());
    F::mul(p2.y(), {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u},
           F::r_sqr());
    C::from_affine(p2);

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2, ctx);
        return &sum;
    };
}

TEST_CASE("jacobian scaler_mul", "[curve][jacobian][scaler_mul]") {
    using C = Dlp1CurveJ;
    using F = Dlp1Field;
    using S = Dlp1Scaler;
    S sOne(1);

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C::Context<> ctx;

    C p;
    F::mul(p.x(),
           {0x1e3b0742u, 0xebf7d73fu, 0xf1a78116u, 0x4c46739au, 0x153663f3u},
           F::r_sqr());
    F::mul(p.y(),
           {0x16a8c9aau, 0xc4ad5fdfu, 0x58163ef3u, 0x9de531f5u, 0xe9cb1575u},
           F::r_sqr());
    C::from_affine(p);
    REQUIRE(C::on_curve(p, ctx));
    CAPTURE(p);

    C prod1, prod2, sum;

    C::mul(prod1, 0, p, ctx);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    C::mul(prod1, 1, p, ctx);
    CAPTURE(prod1);
    REQUIRE(prod1.x() == p.x());
    REQUIRE(prod1.y() == p.y());
    REQUIRE(prod1.z() == p.z());

    C::mul(prod1, S::mod(), p, ctx);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    S s1, s2;
    for (int k = 0; k < 100; ++k) {
        S::sample(s1, rng);
        S::neg(s2, s1);

        C::mul(prod1, s1, p, ctx);
        CAPTURE(prod1);
        C::mul(prod2, s2, p, ctx);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2, ctx);
        CAPTURE(sum);
        REQUIRE(sum.is_inf());

        S::add(s1, s2, sOne);
        C::mul(prod2, s1, p, ctx);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2, ctx);
        CAPTURE(sum);
        REQUIRE(C::eq(sum, p, ctx));
    }
}