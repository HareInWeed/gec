#include <common.hpp>

#include <gec/bigint.hpp>
#include <gec/curve.hpp>

#include <curve.hpp>

#include <configured_catch.hpp>

using namespace gec;
using namespace bigint;
using namespace gec::bigint::literal;
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

    C test(F(1), F(1));
    REQUIRE(!C::on_curve(test));

    C p1;
    F::to_montgomery(p1.x(), //
                     0x0ee27967'5de1bde5'faf553e9'2185fec7'43e7dd56_int);
    F::to_montgomery(p1.y(), //
                     0xa43c088f'a471d05c'3d1bed80'b89428be'84e54fae_int);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1));

    C p2;
    F::to_montgomery(p2.x(), //
                     0x16b60634'e1d3e896'2879d7aa'2c1672ab'de0252bb_int);
    F::to_montgomery(p2.y(), //
                     0x99056d94'e6864afa'a034f181'd8b4192f'1cbedd98_int);
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2));

    C sum, expected;

    C::add(sum, p1, p2);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(
        expected.x(), //
        {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u});
    F::to_montgomery(
        expected.y(), //
        {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    C::add(sum, p1, p1);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(
        expected.x(), //
        {0x6b52f5f8u, 0x836d4559u, 0x4eb4f96fu, 0x11b16271u, 0xb9194d96u});
    F::to_montgomery(
        expected.y(), //
        {0x1fd6f136u, 0xcd8ecae6u, 0xbec3bb77u, 0xa5bdc183u, 0x842648beu});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    C::add(sum, p2, p2);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(
        expected.x(), //
        {0x34aabf2eu, 0xf06c1194u, 0xbd316d0au, 0x3a407ef7u, 0x850f874eu});
    F::to_montgomery(
        expected.y(), //
        {0x1870fd80u, 0xe627d83bu, 0x7af69418u, 0xad073ee5u, 0xba3606e5u});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    p2.set_inf();
    C::add(sum, p1, p2);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    REQUIRE(C::eq(p1, sum));

    p1.set_inf();
    C::add(sum, p1, p2);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    REQUIRE(sum.is_inf());
}

TEST_CASE("affine bench", "[curve][affine][bench]") {
    using C = CurveA2;
    using F = Field160_2;

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
        C::add(sum, p1, p2);
        return &sum;
    };
}

TEST_CASE("projective", "[curve][projective]") {
    using C = CurveP;
    using F = Field160;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           F::r_sqr());
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           F::r_sqr());
    C::from_affine(p1);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           F::r_sqr());
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           F::r_sqr());
    C::from_affine(p2);
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2));

    C sum;
    C::add(sum, p1, p2);
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           F::r_sqr());
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           F::r_sqr());
    C::from_affine(expected);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));

    CHECK(C::eq(expected, sum));
}

TEST_CASE("projective bench", "[curve][projective][bench]") {
    using C = CurveP2;
    using F = Field160_2;

    C p1;
    F::to_montgomery(p1.x(),
                     {0x0ee27967u, 0x5de1bde5faf553e9u, 0x2185fec743e7dd56u});
    F::to_montgomery(p1.y(),
                     {0xa43c088fu, 0xa471d05c3d1bed80u, 0xb89428be84e54faeu});
    C::from_affine(p1);

    C p2;
    F::to_montgomery(p2.x(),
                     {0x16b60634u, 0xe1d3e8962879d7aau, 0x2c1672abde0252bbu});
    F::to_montgomery(p2.y(),
                     {0x99056d94u, 0xe6864afaa034f181u, 0xd8b4192f1cbedd98u});
    C::from_affine(p2);

    C sum;
    BENCHMARK("add") {
        C::add(sum, p1, p2);
        return &sum;
    };
}

TEST_CASE("jacobian", "[curve][jacobian]") {
    using C = CurveJ;
    using F = Field160;

    C p1;
    F::mul(p1.x(),
           {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u},
           F::r_sqr());
    F::mul(p1.y(),
           {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu},
           F::r_sqr());
    C::from_affine(p1);
    REQUIRE(C::on_curve(p1));

    C p2;
    F::mul(p2.x(),
           {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu},
           F::r_sqr());
    F::mul(p2.y(),
           {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u},
           F::r_sqr());
    C::from_affine(p2);
    REQUIRE(C::on_curve(p2));

    C sum;
    C::add(sum, p1, p2);
    REQUIRE(C::on_curve(sum));

    C expected;
    F::mul(expected.x(),
           {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u},
           F::r_sqr());
    F::mul(expected.y(),
           {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu},
           F::r_sqr());
    C::from_affine(expected);
    REQUIRE(C::on_curve(expected));

    CAPTURE(expected, sum);
    CHECK(C::eq(expected, sum));
}

TEST_CASE("jacobian bench", "[curve][jacobian][bench]") {
    using C = CurveJ2;
    using F = Field160_2;

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
        C::add(sum, p1, p2);
        return &sum;
    };
}

TEST_CASE("affine scalar_mul", "[curve][affine][scalar_mul]") {
    using C = Dlp1CurveA;
    using F = Dlp1Field;
    using S = Dlp1Scalar;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C p;
    F::to_montgomery(
        p.x(), //
        {0x1e3b0742u, 0xebf7d73fu, 0xf1a78116u, 0x4c46739au, 0x153663f3u});
    F::to_montgomery(
        p.y(), //
        {0x16a8c9aau, 0xc4ad5fdfu, 0x58163ef3u, 0x9de531f5u, 0xe9cb1575u});
    REQUIRE(C::on_curve(p));
    CAPTURE(p);

    C prod1, prod2, sum;

    C::mul(prod1, 0, p);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    C::mul(prod1, 1, p);
    CAPTURE(prod1);
    REQUIRE(prod1.x() == p.x());
    REQUIRE(prod1.y() == p.y());

    C::mul(prod1, S::mod(), p);
    CAPTURE(prod1);
    REQUIRE(C::on_curve(prod1));
    REQUIRE(prod1.is_inf());

    S s1, s2;
    for (int k = 0; k < 100; ++k) {
        S::sample(s1, rng);
        S::neg(s2, s1);

        C::mul(prod1, s1, p);
        CAPTURE(prod1);
        C::mul(prod2, s2, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(sum.is_inf());

        S::add(s1, s2, 1);
        C::mul(prod2, s1, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(C::eq(sum, p));
    }
}

TEST_CASE("jacobian scalar_mul", "[curve][jacobian][scalar_mul]") {
    using C = Dlp1CurveJ;
    using F = Dlp1Field;
    using S = Dlp1Scalar;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C p;
    F::to_montgomery(
        p.x(), //
        {0x1e3b0742u, 0xebf7d73fu, 0xf1a78116u, 0x4c46739au, 0x153663f3u});
    F::to_montgomery(
        p.y(), //
        {0x16a8c9aau, 0xc4ad5fdfu, 0x58163ef3u, 0x9de531f5u, 0xe9cb1575u});
    C::from_affine(p);
    REQUIRE(C::on_curve(p));
    CAPTURE(p);

    C prod1, prod2, sum;

    C::mul(prod1, 0, p);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    C::mul(prod1, 1, p);
    CAPTURE(prod1);
    REQUIRE(prod1.x() == p.x());
    REQUIRE(prod1.y() == p.y());
    REQUIRE(prod1.z() == p.z());

    C::mul(prod1, S::mod(), p);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    S s1, s2;
    for (int k = 0; k < 100; ++k) {
        S::sample(s1, rng);
        S::neg(s2, s1);

        C::mul(prod1, s1, p);
        CAPTURE(prod1);
        C::mul(prod2, s2, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(sum.is_inf());

        S::add(s1, s2, 1);
        C::mul(prod2, s1, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(C::eq(sum, p));
    }
}