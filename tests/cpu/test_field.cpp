#include <common.hpp>
#include <field.hpp>

#include <configured_catch.hpp>

using namespace gec;
using namespace bigint;
using namespace gec::bigint::literal;

TEST_CASE("add group neg", "[add_group][field]") {
    using F = Field160;
    F e;
    F::neg(e, F());
    REQUIRE(e.is_zero());

    F::neg(e, F(0x1));
    REQUIRE(F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int) == e);

    F::neg(e, F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int));
    REQUIRE(F(0x1_int) == e);

    F::neg(e, F(0x5bbc8155'ec6dcb13'faebe765'2e0bf7b6'2f1d84b4_int));
    REQUIRE(F(0x5bbc8155'ec6dcb13'faebe765'2e0bf7b6'2f1d84b5_int) == e);
}

TEST_CASE("add group add", "[add_group][field]") {
    using F = Field160;
    F e;

    F::add(e, F(), F());
    REQUIRE(e.is_zero());

    F::add(e, F(1), F(2));
    REQUIRE(F(3) == e);

    F::add(e, F(0x2), F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0966_int));
    REQUIRE(F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int) == e);

    F::add(e, F(0x2), F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int));
    REQUIRE(F(0x1) == e);

    F::add(e, F(0x0d1f4b5b'8005d7aa'4fed62ac'03831479'83ccd32d_int),
           F(0x1cfaec75'7faf7c19'd3121b9e'ded3ca3b'952e1b38_int));
    REQUIRE(F(0x2a1a37d0'ffb553c4'22ff7e4a'e256deb5'18faee65_int) == e);

    F::add(e, F(0x8f566078'b1d6a8df'd5af7fad'aa89f612'240a6b52_int),
           F(0x4a617461'4c8165c6'f378a372'8d6cccb6'd07f7850_int));
    REQUIRE(F(0x223ed22e'257c787e'd3505455'dbded35c'964eda39_int) == e);
}

TEST_CASE("add group sub", "[add_group][field]") {
    using F = Field160;
    F e;

    F::sub(e, F(), F());
    REQUIRE(e.is_zero());

    F::sub(e, F(0xf0), F(0x2));
    REQUIRE(F(0xee) == e);

    F::sub(e, F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int),
           F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0966_int));
    REQUIRE(F(0x2) == e);

    F::sub(e, F(0x1), F(0x2));
    REQUIRE(F(0xb77902ab'd8db9627'f5d7ceca'5c17ef6c'5e3b0968_int) == e);

    F::sub(e, F(0x2a1a37d0'ffb553c4'22ff7e4a'e256deb5'18faee65_int),
           F(0x1cfaec75'7faf7c19'd3121b9e'ded3ca3b'952e1b38_int));
    REQUIRE(F(0x0d1f4b5b'8005d7aa'4fed62ac'03831479'83ccd32d_int) == e);

    F::sub(e, F(0x223ed22e'257c787e'd3505455'dbded35c'964eda39_int),
           F(0x4a617461'4c8165c6'f378a372'8d6cccb6'd07f7850_int));
    REQUIRE(F(0x8f566078'b1d6a8df'd5af7fad'aa89f612'240a6b52_int) == e);
}

TEST_CASE("mul_pow2", "[add_group][field]") {
    using F = Field160;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    F a, a2, a4, a8, res;
    do {
        for (size_t i = 0; i < LN_160; ++i) {
            a.array()[i] = dis_u32(gen);
        }
    } while (a >= F::mod());
    F::add(a2, a, a);
    F::add(a4, a2, a2);
    F::add(a8, a4, a4);
    CAPTURE(a, a2, a4);

    res = a;
    F::add_self(res);
    REQUIRE(a2 == res);

    res = a;
    F::mul_pow2<1>(res);
    REQUIRE(a2 == res);

    res = a;
    F::mul_pow2<2>(res);
    REQUIRE(a4 == res);

    res = a;
    F::mul_pow2<3>(res);
    REQUIRE(a8 == res);
}

using SmallArray = ArrayBE<LIMB_T, 3>;
GEC_DEF(SmallMod, static const SmallArray, 0x0, 0xb, 0x7);

TEST_CASE("random sampling", "[add_group][field][random]") {
    using F1 = Field160;
    using F2 = Field160_2;
    using G = GEC_BASE_ADD_GROUP(SmallArray, SmallMod);

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

#define test(Int)                                                              \
    do {                                                                       \
        Int x, y, z;                                                           \
        for (int k = 0; k < 10000; ++k) {                                      \
            Int::sample(x, rng);                                               \
            REQUIRE(x < Int::mod());                                           \
                                                                               \
            Int::sample_non_zero(x, rng);                                      \
            REQUIRE(!x.is_zero());                                             \
            REQUIRE(x < Int::mod());                                           \
                                                                               \
            Int::sample(y, x, rng);                                            \
            REQUIRE(y < x);                                                    \
                                                                               \
            Int::sample(z, y, x, rng);                                         \
            REQUIRE(z < x);                                                    \
            REQUIRE(y <= z);                                                   \
                                                                               \
            Int::sample_inclusive(z, x, rng);                                  \
            REQUIRE(z <= x);                                                   \
                                                                               \
            Int::sample_inclusive(z, y, x, rng);                               \
            REQUIRE(z <= x);                                                   \
            REQUIRE(y <= z);                                                   \
        }                                                                      \
    } while (false)

    test(F1);
    test(F2);
    test(G);

#undef test
}

TEST_CASE("mul_pow2 bench", "[add_group][bench]") {
    using F = Field160;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    F a;
    do {
        for (size_t i = 0; i < LN_160; ++i) {
            a.array()[i] = dis_u32(gen);
        }
    } while (a >= F::mod());

    BENCHMARK("add to 2a") {
        F res = a;
        F::add(res, a, a);
        return res;
    };

    BENCHMARK("add to 4a") {
        F res, tmp;
        F::add(tmp, a, a);
        F::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("add to 8a") {
        F res, tmp;
        F::add(res, a, a);
        F::add(tmp, res, res);
        F::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("add to 2^32 a") {
        F res, tmp;
        F::add(tmp, a, a);
        for (int k = 0; k < 31; ++k) {
            F::add(res, tmp, tmp);
            F::add(tmp, res, res);
        }
        F::add(res, tmp, tmp);
        return res;
    };

    BENCHMARK("mul 2") {
        F res = a;
        F::mul_pow2<1>(res);
        return res;
    };

    BENCHMARK("mul 4") {
        F res = a;
        F::mul_pow2<2>(res);
        return res;
    };

    BENCHMARK("mul 8") {
        F res = a;
        F::mul_pow2<3>(res);
        return res;
    };

    BENCHMARK("mul 2^32") {
        F res = a;
        F::mul_pow2<32>(res);
        return res;
    };
}

TEST_CASE("montgomery mul", "[ring][field]") {
    using F = Field160;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    F a, b;

    F::to_montgomery(a, F());
    REQUIRE(F(0) == a);

    F::from_montgomery(b, a);
    REQUIRE(F(0) == b);

    F::to_montgomery(a, F(0xffffffff_int));
    REQUIRE(F(0xad37b410'255c6eb2'7601a883'659883e8'070707fc_int) == a);

    F::from_montgomery(b, a);
    REQUIRE(F(0xffffffff_int) == b);

    F c, d, e;
    do {
        for (size_t i = 0; i < LN_160; ++i) {
            c.array()[i] = dis_u32(gen);
        }
    } while (c >= F::mod());

    d = c;
    F::to_montgomery(e, d);
    F::from_montgomery(d, e);
    REQUIRE(c == d);

    LIMB_T l, h, x, y;
    F mon_x, mon_y, mon_xy, xy;

    x = 0xd8b2f21eu;
    y = 0xabf7c642u;
    utils::uint_mul_lh(l, h, x, y);
    F::to_montgomery(mon_x, F(x));
    F::to_montgomery(mon_y, F(y));
    F::mul(mon_xy, mon_x, mon_y);
    F::from_montgomery(xy, mon_xy);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    F::to_montgomery(mon_x, F(x));
    F::to_montgomery(mon_y, F(y));
    F::mul(mon_xy, mon_x, mon_y);
    F::from_montgomery(xy, mon_xy);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    F::to_montgomery(mon_x, F(x));
    F::to_montgomery(mon_y, F(y));
    F::mul(mon_xy, mon_x, mon_y);
    F::from_montgomery(xy, mon_xy);
    REQUIRE(l == xy.array()[0]);
    REQUIRE(h == xy.array()[1]);

    mon_x = F(0xa5481e14'293b3c7d'b85ecae1'83d79492'cd652763_int);
    mon_y = F(0x93d20f51'898541bb'74aa1184'bccb10b2'47f79c2c_int);
    F::mul(mon_xy, mon_x, mon_y);
    REQUIRE(F(0x4886fd54'272469d8'0a283135'a3e81093'a1c4f697_int) == mon_xy);
}

#ifdef GEC_ENABLE_AVX2

TEST_CASE("avx2 montgomery", "[ring][avx2]") {
    using gec::utils::CmpEnum;
    using gec::utils::VtSeqCmp;
    using Int = GEC_BASE_ADD_GROUP(Array256, MOD_256);

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    Array256 x_arr(
        0x1f82f372'62639538'ca640ff9'ed12396a'9c4d50da'ff21e339'fbfa64d8'75b40000_int);
    Array256 y_arr(
        0xed469d79'aba8d6fa'6724432c'7221f040'6416351d'923ec2ca'72bc1127'f1e018aa_int);
    Int &x_int = static_cast<Int &>(x_arr);
    Int &y_int = static_cast<Int &>(y_arr);
    Array256 mon_x_arr, mon_y_arr, mon_xy_arr, xy_arr;

    for (int k = 0; k < 10000; ++k) {
        Int::sample(x_int, rng);
        Int::sample(y_int, rng);
        CAPTURE(Int::mod(), x_int, y_int);

        {
            using F =
                GEC_BASE_FIELD(Array256, MOD_256, MOD_P_256, RR_256, OneR_256);
            const auto &x = static_cast<F &>(x_arr);
            const auto &y = static_cast<F &>(y_arr);
            auto &mon_x = static_cast<F &>(mon_x_arr);
            auto &mon_y = static_cast<F &>(mon_y_arr);
            auto &mon_xy = static_cast<F &>(mon_xy_arr);
            auto &xy = static_cast<F &>(xy_arr);

            F::to_montgomery(mon_x, x);
            F::to_montgomery(mon_y, y);
            F::mul(mon_xy, mon_x, mon_y);
            F::from_montgomery(xy, mon_xy);
        }
        CAPTURE(mon_x_arr, mon_y_arr);
        CAPTURE(mon_xy_arr, xy_arr);

        {
            using F = GEC_BASE_AVX2FIELD(Array256, MOD_256, MOD_P_256, RR_256,
                                         OneR_256);
            const auto &x = static_cast<F &>(x_arr);
            const auto &y = static_cast<F &>(y_arr);
            const auto &expected_mon_x = static_cast<F &>(mon_x_arr);
            const auto &expected_mon_y = static_cast<F &>(mon_y_arr);
            const auto &expected_mon_xy = static_cast<F &>(mon_xy_arr);
            const auto &expected_xy = static_cast<F &>(xy_arr);

            F mon_x, mon_y, mon_xy, xy;
            F::to_montgomery(mon_x, x);
            CAPTURE(mon_x);
            REQUIRE(expected_mon_x == mon_x);
            F::to_montgomery(mon_y, y);
            CAPTURE(mon_y);
            REQUIRE(expected_mon_y == mon_y);
            F::mul(mon_xy, mon_x, mon_y);
            CAPTURE(mon_xy);
            REQUIRE(expected_mon_xy == mon_xy);
            F::from_montgomery(xy, mon_xy);
            CAPTURE(xy);
            REQUIRE(expected_xy == xy);
        }
    }
}

TEST_CASE("256 montgomery bench", "[ring][avx2][bench]") {
    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));
    using Int = GEC_BASE_ADD_GROUP(Array256, MOD_256);
    using SerialF =
        GEC_BASE_FIELD(Array256, MOD_256, MOD_P_256, RR_256, OneR_256);
    using AVX2F =
        GEC_BASE_AVX2FIELD(Array256, MOD_256, MOD_P_256, RR_256, OneR_256);

    Array256 x_arr, y_arr, mon_x_arr, mon_y_arr;
    Int &x_int = static_cast<Int &>(x_arr);
    Int &y_int = static_cast<Int &>(y_arr);

    Int::sample(x_int, rng);
    Int::sample(y_int, rng);

    {
        using F = SerialF;
        const auto &x = static_cast<const F &>(x_arr);
        const auto &y = static_cast<const F &>(y_arr);
        auto &mon_x = static_cast<F &>(mon_x_arr);
        auto &mon_y = static_cast<F &>(mon_y_arr);

        F::to_montgomery(mon_x, x);
        F::to_montgomery(mon_y, y);
    }

    {
        using F = SerialF;
        const auto &x = static_cast<const F &>(x_arr);
        const auto &mon_x = static_cast<const F &>(mon_x_arr);
        const auto &mon_y = static_cast<const F &>(mon_y_arr);

        BENCHMARK("into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
            return res;
        };

        BENCHMARK("montgomery mul") {
            F mon_xy;
            F::mul(mon_xy, mon_x, mon_y);
            return mon_xy;
        };
    }

    {
        using F = AVX2F;
        const auto &x = static_cast<const F &>(x_arr);
        const auto &mon_x = static_cast<const F &>(mon_x_arr);
        const auto &mon_y = static_cast<const F &>(mon_y_arr);

        BENCHMARK("avx2 into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("avx2 from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
            return res;
        };

        BENCHMARK("avx2 montgomery mul") {
            F mon_xy;
            F::mul(mon_xy, mon_x, mon_y);
            return mon_xy;
        };
    }
}

#endif // GEC_ENABLE_AVX2

TEST_CASE("montgomery inv", "[field]") {
    using F = Field160;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    F a, mon_a, inv_a, mon_prod, prod;
    for (int k = 0; k < 10000; ++k) {
        do {
            for (size_t k = 0; k < LN_160; ++k) {
                a.array()[k] = dis_u32(gen);
            }
        } while (a >= F::mod());
        // a = Field(0x31a50ad6'93f524b7'a6ea2efe'ed31237a_int,
        //           0x2d2731f7_int);
        F::to_montgomery(mon_a, a);
        F::inv(inv_a, mon_a);
        F::mul(mon_prod, mon_a, inv_a);
        F::from_montgomery(prod, mon_prod);
        CAPTURE(prod);
        REQUIRE(prod.is_one());
    }
}

TEST_CASE("montgomery exp", "[field]") {
    using F = Field160;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    F mod_m, a, mon_a, mon_exp_a, exp_a;
    F::sub(mod_m, F::mod(), 1);

    for (int k = 0; k < 10000; ++k) {
        do {
            for (size_t k = 0; k < LN_160; ++k) {
                a.array()[k] = dis_u32(gen);
            }
        } while (a >= F::mod() && !a.is_zero());
        // a = Field(0x31a50ad6'93f524b7'a6ea2efe'ed31237a_int,
        //           0x2d2731f7_int);
        F::to_montgomery(mon_a, a);

        F::pow(mon_exp_a, mon_a, 1);
        REQUIRE(mon_exp_a == mon_a);

        F::pow(mon_exp_a, mon_a, 0);
        F::from_montgomery(exp_a, mon_exp_a);
        REQUIRE(exp_a.is_one());

        F::pow(mon_exp_a, mon_a, F::mod());
        REQUIRE(mon_exp_a == mon_a); // Fermat's Little Theorem

        F::pow(mon_exp_a, mon_a, mod_m);
        F::from_montgomery(exp_a, mon_exp_a);
        REQUIRE(exp_a.is_one()); // Fermat's Little Theorem
    }
}

TEST_CASE("montgomery mul bench", "[ring][field][bench]") {
    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

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
    } while (x0 >= Field160_2::mod());
    do {
        y0.array()[0] = dis_u64(gen);
        y0.array()[1] = dis_u64(gen);
        y0.array()[2] = dis_u32(gen);
    } while (y0 >= Field160_2::mod());
    Field160_2::to_montgomery(mon_x0, x0);
    Field160_2::to_montgomery(mon_y0, y0);

    {
        using F = Field160;
        const F &x = *reinterpret_cast<const F *>(x0.array());
        const F &mon_x = *reinterpret_cast<const F *>(mon_x0.array());
        const F &mon_y = *reinterpret_cast<const F *>(mon_y0.array());

        BENCHMARK("32-bits into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("32-bits from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
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
        const F &x = reinterpret_cast<const F &>(x0);
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        const F &mon_y = reinterpret_cast<const F &>(mon_y0);

        BENCHMARK("64-bits into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("64-bits from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
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
    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    std::mt19937 gen(seed);

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
    } while (x0 >= Field160_2::mod());
    Field160_2::mul(mon_x0, x0, Field160_2::r_sqr());

    {
        using F = Field160;
        const F &mon_x = *reinterpret_cast<const F *>(mon_x0.array());
        BENCHMARK("32-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x);
            return inv_x;
        };
    }

    {
        using F = Field160_2;
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        BENCHMARK("64-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x);
            return inv_x;
        };
    }
}

template <typename F>
static void test_mod_sqrt(std::random_device::result_type seed) {
    CAPTURE(seed);

    auto rng = make_gec_rng(std::mt19937(seed));
    F x, xx, sqrt, sqr;
    for (int k = 0; k < 1000; ++k) {
        F::sample(x, rng);
        F::mul(xx, x, x);
        CAPTURE(x, xx);
        REQUIRE(F::mod_sqrt(sqrt, xx, rng));
        CAPTURE(sqrt);
        F::mul(sqr, sqrt, sqrt);
        REQUIRE(xx == sqr);
    }
}
TEST_CASE("montgomery mod_sqrt", "[field][quadratic_residue]") {
    std::random_device rd;
    test_mod_sqrt<Field160>(rd());
    test_mod_sqrt<Field160_2>(rd());
}

TEST_CASE("bigint hash", "[field][hash]") {
    using F = Field160;
    F::Hasher h;

    F Zero(0), One(1);

    REQUIRE(h(Zero) != h(One));
}