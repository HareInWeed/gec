#include "common.hpp"
#include "field.hpp"

#include <gec/utils.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

TEST_CASE("add group neg", "[add_group][field]") {
    Field160 e;
    Field160::neg(e, Field160());
    REQUIRE(e.is_zero());

    Field160::neg(e, Field160(0x1u));
    REQUIRE(Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                     0x5e3b0968u) == e);

    Field160::neg(e, Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau,
                              0x5c17ef6cu, 0x5e3b0968u));
    REQUIRE(Field160(0x1u) == e);

    Field160::neg(e, Field160(0x5bbc8155u, 0xec6dcb13u, 0xfaebe765u,
                              0x2e0bf7b6u, 0x2f1d84b4u));
    REQUIRE(Field160(0x5bbc8155u, 0xec6dcb13u, 0xfaebe765u, 0x2e0bf7b6u,
                     0x2f1d84b5u) == e);
}

TEST_CASE("add group add", "[add_group][field]") {
    Field160 e;

    Field160::add(e, Field160(), Field160());
    REQUIRE(e.is_zero());

    Field160::add(e, Field160(1), Field160(2));
    REQUIRE(Field160(3) == e);

    Field160::add(e, Field160(0x2),
                  Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                           0x5e3b0966u));
    REQUIRE(Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                     0x5e3b0968u) == e);

    Field160::add(e, Field160(0x2),
                  Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                           0x5e3b0968u));
    REQUIRE(Field160(0x1) == e);

    Field160::add(e,
                  Field160(0x0d1f4b5bu, 0x8005d7aau, 0x4fed62acu, 0x03831479u,
                           0x83ccd32du),
                  Field160(0x1cfaec75u, 0x7faf7c19u, 0xd3121b9eu, 0xded3ca3bu,
                           0x952e1b38u));
    REQUIRE(Field160(0x2a1a37d0u, 0xffb553c4u, 0x22ff7e4au, 0xe256deb5u,
                     0x18faee65u) == e);

    Field160::add(e,
                  Field160(0x8f566078u, 0xb1d6a8dfu, 0xd5af7fadu, 0xaa89f612u,
                           0x240a6b52u),
                  Field160(0x4a617461u, 0x4c8165c6u, 0xf378a372u, 0x8d6cccb6u,
                           0xd07f7850u));
    REQUIRE(Field160(0x223ed22eu, 0x257c787eu, 0xd3505455u, 0xdbded35cu,
                     0x964eda39u) == e);
}

TEST_CASE("add group sub", "[add_group][field]") {
    Field160 e;

    Field160::sub(e, Field160(), Field160());
    REQUIRE(e.is_zero());

    Field160::sub(e, Field160(0xf0), Field160(0x2));
    REQUIRE(Field160(0xee) == e);

    Field160::sub(e,
                  Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                           0x5e3b0968u),
                  Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                           0x5e3b0966u));
    REQUIRE(Field160(0x2) == e);

    Field160::sub(e, Field160(0x1), Field160(0x2));
    REQUIRE(Field160(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                     0x5e3b0968u) == e);

    Field160::sub(e,
                  Field160(0x2a1a37d0u, 0xffb553c4u, 0x22ff7e4au, 0xe256deb5u,
                           0x18faee65u),
                  Field160(0x1cfaec75u, 0x7faf7c19u, 0xd3121b9eu, 0xded3ca3bu,
                           0x952e1b38u));
    REQUIRE(Field160(0x0d1f4b5bu, 0x8005d7aau, 0x4fed62acu, 0x03831479u,
                     0x83ccd32du) == e);

    Field160::sub(e,
                  Field160(0x223ed22eu, 0x257c787eu, 0xd3505455u, 0xdbded35cu,
                           0x964eda39u),
                  Field160(0x4a617461u, 0x4c8165c6u, 0xf378a372u, 0x8d6cccb6u,
                           0xd07f7850u));
    REQUIRE(Field160(0x8f566078u, 0xb1d6a8dfu, 0xd5af7fadu, 0xaa89f612u,
                     0x240a6b52u) == e);
}

TEST_CASE("mul_pow2", "[add_group][field]") {
    const Field160 &Mod = reinterpret_cast<const Field160 &>(MOD_160);
    const Field160 &RR = reinterpret_cast<const Field160 &>(RR_160);
    const Field160 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field160 a, a2, a4, a8, res;
    do {
        for (int i = 0; i < LN_160; ++i) {
            a.array()[i] = dis_u32(gen);
        }
    } while (a >= Mod);
    Field160::add(a2, a, a);
    Field160::add(a4, a2, a2);
    Field160::add(a8, a4, a4);
    CAPTURE(a, a2, a4);

    res = a;
    Field160::add_self(res);
    REQUIRE(a2 == res);

    res = a;
    Field160::mul_pow2<1>(res);
    REQUIRE(a2 == res);

    res = a;
    Field160::mul_pow2<2>(res);
    REQUIRE(a4 == res);

    res = a;
    Field160::mul_pow2<3>(res);
    REQUIRE(a8 == res);
}

TEST_CASE("mul_pow2 bench", "[add_group][bench]") {
    const Field160 &Mod = reinterpret_cast<const Field160 &>(MOD_160);
    const Field160 &RR = reinterpret_cast<const Field160 &>(RR_160);
    const Field160 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field160 a;
    do {
        for (int i = 0; i < LN_160; ++i) {
            a.array()[i] = dis_u32(gen);
        }
    } while (a >= Mod);

    BENCHMARK("add to 2a") {
        Field160 res = a;
        Field160::add(res, a, a);
        return res;
    };

    BENCHMARK("add to 4a") {
        Field160 res, tmp;
        Field160::add(tmp, a, a);
        Field160::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("add to 8a") {
        Field160 res, tmp;
        Field160::add(res, a, a);
        Field160::add(tmp, res, res);
        Field160::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("add to 2^32 a") {
        Field160 res, tmp;
        Field160::add(tmp, a, a);
        for (int k = 0; k < 31; ++k) {
            Field160::add(res, tmp, tmp);
            Field160::add(tmp, res, res);
        }
        Field160::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("mul 2") {
        Field160 res = a;
        Field160::mul_pow2<1>(res);
        return res;
    };

    BENCHMARK("mul 4") {
        Field160 res = a;
        Field160::mul_pow2<2>(res);
        return res;
    };

    BENCHMARK("mul 8") {
        Field160 res = a;
        Field160::mul_pow2<3>(res);
        return res;
    };

    BENCHMARK("mul 2^32") {
        Field160 res = a;
        Field160::mul_pow2<32>(res);
        return res;
    };
}

TEST_CASE("montgomery mul", "[ring][field]") {
    const Field160 &Mod = reinterpret_cast<const Field160 &>(MOD_160);
    const Field160 &RR = reinterpret_cast<const Field160 &>(RR_160);
    const Field160 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field160 a, b;

    Field160::mul(a, Field160(), RR);
    REQUIRE(Field160(0) == a);

    Field160::mul(b, a, One);
    REQUIRE(Field160(0) == b);

    Field160::mul(a, Field160(0xffffffffu), RR);
    REQUIRE(Field160(0xad37b410u, 0x255c6eb2u, 0x7601a883u, 0x659883e8u,
                     0x070707fcu) == a);

    Field160::mul(b, a, One);
    REQUIRE(Field160(0xffffffffu) == b);

    Field160 c, d, e;
    do {
        for (int i = 0; i < LN_160; ++i) {
            c.array()[i] = dis_u32(gen);
        }
    } while (c >= Mod);

    d = c;
    Field160::mul(e, d, RR);
    Field160::mul(d, e, One);
    REQUIRE(c == d);

    LIMB_T l, h, x, y;
    Field160 mon_x, mon_y, mon_xy, xy;

    x = 0xd8b2f21eu;
    y = 0xabf7c642u;
    utils::uint_mul_lh(l, h, x, y);
    Field160::mul(mon_x, Field160(x), RR);
    Field160::mul(mon_y, Field160(y), RR);
    Field160::mul(mon_xy, mon_x, mon_y);
    Field160::mul(xy, mon_xy, One);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    Field160::mul(mon_x, Field160(x), RR);
    Field160::mul(mon_y, Field160(y), RR);
    Field160::mul(mon_xy, mon_x, mon_y);
    Field160::mul(xy, mon_xy, One);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    Field160::mul(mon_x, RR, Field160(x));
    Field160::mul(mon_y, RR, Field160(y));
    Field160::mul(mon_xy, mon_x, mon_y);
    Field160::mul(xy, One, mon_xy);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    mon_x = Field160(0xa5481e14u, 0x293b3c7du, 0xb85ecae1u, 0x83d79492u,
                     0xcd652763u);
    mon_y = Field160(0x93d20f51u, 0x898541bbu, 0x74aa1184u, 0xbccb10b2u,
                     0x47f79c2cu);
    Field160::mul(mon_xy, mon_x, mon_y);
    REQUIRE(Field160(0x4886fd54u, 0x272469d8u, 0x0a283135u, 0xa3e81093u,
                     0xa1c4f697u) == mon_xy);
}

TEST_CASE("montgomery inv", "[field]") {
    Field160::Context ctx;
    const auto &RR = reinterpret_cast<const Field160 &>(RR_160);
    const Field160 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field160 a, mon_a, inv_a, mon_prod, prod;
    for (int k = 0; k < 10000; ++k) {
        do {
            for (int k = 0; k < LN_160; ++k) {
                a.array()[k] = dis_u32(gen);
            }
        } while (a >= RR);
        // a = Field(0x31a50ad6u, 0x93f524b7u, 0xa6ea2efeu, 0xed31237au,
        //           0x2d2731f7u);
        Field160::mul(mon_a, a, RR);
        Field160::inv(inv_a, mon_a, ctx);
        Field160::mul(mon_prod, mon_a, inv_a);
        Field160::mul(prod, mon_prod, One);
        REQUIRE(One == prod);
    }
}

TEST_CASE("montgomery exp", "[field]") {
    const auto &RR = reinterpret_cast<const Field160 &>(RR_160);
    const auto &Mod = reinterpret_cast<const Field160 &>(MOD_160);
    const Field160 Zero, One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field160::Context ctx;
    Field160 mod_m, a, mon_a, mon_exp_a, mon_prod, exp_a;
    Field160::sub(mod_m, Mod, One);

    for (int k = 0; k < 10000; ++k) {
        do {
            for (int k = 0; k < LN_160; ++k) {
                a.array()[k] = dis_u32(gen);
            }
        } while (a >= RR && !a.is_zero());
        // a = Field(0x31a50ad6u, 0x93f524b7u, 0xa6ea2efeu, 0xed31237au,
        //           0x2d2731f7u);
        Field160::mul(mon_a, a, RR);

        Field160::pow(mon_exp_a, mon_a, 1u, ctx);
        REQUIRE(mon_exp_a == mon_a);

        Field160::pow(mon_exp_a, mon_a, 0u, ctx);
        Field160::mul(exp_a, mon_exp_a, One);
        REQUIRE(One == exp_a);

        Field160::pow(mon_exp_a, mon_a, Mod, ctx);
        REQUIRE(mon_exp_a == mon_a); // Fermat's Little Theorem

        Field160::pow(mon_exp_a, mon_a, mod_m, ctx);
        Field160::mul(exp_a, mon_exp_a, One);
        REQUIRE(One == exp_a); // Fermat's Little Theorem
    }
}

TEST_CASE("montgomery mul bench", "[ring][field][bench]") {
    const auto &MOD = reinterpret_cast<const Field160_2 &>(MOD2_160);
    const auto &RR = reinterpret_cast<const Field160_2 &>(RR2_160);
    const Field160_2 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());
    std::uniform_int_distribution<LIMB2_T> dis_u64(
        std::numeric_limits<LIMB2_T>::min(),
        std::numeric_limits<LIMB2_T>::max());
    Field160_2 x0, y0, mon_x0, mon_y0;
    do {
        x0.array()[0] = dis_u64(gen);
        x0.array()[1] = dis_u64(gen);
        x0.array()[2] = dis_u32(gen);
    } while (x0 >= MOD);
    do {
        y0.array()[0] = dis_u64(gen);
        y0.array()[1] = dis_u64(gen);
        y0.array()[2] = dis_u32(gen);
    } while (y0 >= MOD);
    Field160_2::mul(mon_x0, x0, RR);
    Field160_2::mul(mon_y0, y0, RR);

    {
        using F = Field160;
        const F &RR = reinterpret_cast<const F &>(RR_160);
        const F One(1);
        const F &x = reinterpret_cast<const F &>(x0);
        const F &y = reinterpret_cast<const F &>(y0);
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        const F &mon_y = reinterpret_cast<const F &>(mon_y0);

        BENCHMARK("32-bits into montgomery form") {
            F res;
            F::mul(res, x, RR);
            return res;
        };

        BENCHMARK("32-bits from montgomery form") {
            F res;
            F::mul(res, mon_x, One);
            return res;
        };

        BENCHMARK("32-bits montgomery mul") {
            F xy;
            F::mul(xy, mon_x, mon_y);
            return xy;
        };
    }

    {
        using F = Field160_2;
        const F &RR = reinterpret_cast<const F &>(RR2_160);
        const F One(1);
        const F &x = reinterpret_cast<const F &>(x0);
        const F &y = reinterpret_cast<const F &>(y0);
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        const F &mon_y = reinterpret_cast<const F &>(mon_y0);

        BENCHMARK("64-bits into montgomery form") {
            F res;
            F::mul(res, x, RR);
            return res;
        };

        BENCHMARK("64-bits from montgomery form") {
            F res;
            F::mul(res, mon_x, One);
            return res;
        };

        BENCHMARK("64-bits montgomery mul") {
            F xy;
            F::mul(xy, mon_x, mon_y);
            return xy;
        };
    }
}

TEST_CASE("montgomery inv bench", "[field][bench]") {
    const auto &MOD = reinterpret_cast<const Field160_2 &>(MOD2_160);
    const auto &RR = reinterpret_cast<const Field160_2 &>(RR2_160);
    const Field160_2 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());
    std::uniform_int_distribution<LIMB2_T> dis_u64(
        std::numeric_limits<LIMB2_T>::min(),
        std::numeric_limits<LIMB2_T>::max());
    Field160_2 x0, mon_x0;
    do {
        x0.array()[0] = dis_u64(gen);
        x0.array()[1] = dis_u64(gen);
        x0.array()[2] = dis_u32(gen);
    } while (x0 >= MOD);
    Field160_2::mul(mon_x0, x0, RR);

    {
        using F = Field160;
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        F::Context ctx;
        BENCHMARK("32-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x, ctx);
            return inv_x;
        };
    }

    {
        using F = Field160_2;
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        F::Context ctx;
        BENCHMARK("64-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x, ctx);
            return inv_x;
        };
    }
}
