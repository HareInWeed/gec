#include "common.hpp"

#include <gec/bigint.hpp>
#include <gec/bigint/mixin/ostream.hpp>
#include <gec/bigint/mixin/print.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

class Field
    : public Array<LIMB_T, LN_160>,
      public VtCompareMixin<Field, LIMB_T, LN_160>,
      public BitOpsMixin<Field, LIMB_T, LN_160>,
      public ModAddSubMixin<Field, LIMB_T, LN_160, MOD_160>,
      public Montgomery<Field, LIMB_T, LN_160, MOD_160, MOD_P_160, RR_160>,
      public ArrayOstreamMixin<Field, LIMB_T, LN_160>,
      public ArrayPrintMixin<Field, LIMB_T, LN_160> {
  public:
    using Array::Array;
};

class Field2 : public Array<LIMB2_T, LN2_160>,
               public VtCompareMixin<Field2, LIMB2_T, LN2_160>,
               public BitOpsMixin<Field2, LIMB2_T, LN2_160>,
               public ModAddSubMixin<Field2, LIMB2_T, LN2_160, MOD2_160>,
               public MontgomeryCarryFree<Field2, LIMB2_T, LN2_160, MOD2_160,
                                          MOD2_P_160, RR2_160>,
               public ArrayOstreamMixin<Field2, LIMB2_T, LN2_160>,
               public ArrayPrintMixin<Field2, LIMB2_T, LN2_160> {
  public:
    using Array::Array;
};

TEST_CASE("add group neg", "[add_group][field]") {
    Field e;
    Field::neg(e, Field());
    REQUIRE(e.is_zero());

    Field::neg(e, Field(0x1u));
    REQUIRE(Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                  0x5e3b0968u) == e);

    Field::neg(e, Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                        0x5e3b0968u));
    REQUIRE(Field(0x1u) == e);

    Field::neg(e, Field(0x5bbc8155u, 0xec6dcb13u, 0xfaebe765u, 0x2e0bf7b6u,
                        0x2f1d84b4u));
    REQUIRE(Field(0x5bbc8155u, 0xec6dcb13u, 0xfaebe765u, 0x2e0bf7b6u,
                  0x2f1d84b5u) == e);
}

TEST_CASE("add group add", "[add_group][field]") {
    Field e;

    Field::add(e, Field(), Field());
    REQUIRE(e.is_zero());

    Field::add(e, Field(1), Field(2));
    REQUIRE(Field(3) == e);

    Field::add(
        e, Field(0x2),
        Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu, 0x5e3b0966u));
    REQUIRE(Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                  0x5e3b0968u) == e);

    Field::add(
        e, Field(0x2),
        Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu, 0x5e3b0968u));
    REQUIRE(Field(0x1) == e);

    Field::add(
        e,
        Field(0x0d1f4b5bu, 0x8005d7aau, 0x4fed62acu, 0x03831479u, 0x83ccd32du),
        Field(0x1cfaec75u, 0x7faf7c19u, 0xd3121b9eu, 0xded3ca3bu, 0x952e1b38u));
    REQUIRE(Field(0x2a1a37d0u, 0xffb553c4u, 0x22ff7e4au, 0xe256deb5u,
                  0x18faee65u) == e);

    Field::add(
        e,
        Field(0x8f566078u, 0xb1d6a8dfu, 0xd5af7fadu, 0xaa89f612u, 0x240a6b52u),
        Field(0x4a617461u, 0x4c8165c6u, 0xf378a372u, 0x8d6cccb6u, 0xd07f7850u));
    REQUIRE(Field(0x223ed22eu, 0x257c787eu, 0xd3505455u, 0xdbded35cu,
                  0x964eda39u) == e);
}

TEST_CASE("add group sub", "[add_group][field]") {
    Field e;

    Field::sub(e, Field(), Field());
    REQUIRE(e.is_zero());

    Field::sub(e, Field(0xf0), Field(0x2));
    REQUIRE(Field(0xee) == e);

    Field::sub(
        e,
        Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu, 0x5e3b0968u),
        Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu, 0x5e3b0966u));
    REQUIRE(Field(0x2) == e);

    Field::sub(e, Field(0x1), Field(0x2));
    REQUIRE(Field(0xb77902abu, 0xd8db9627u, 0xf5d7cecau, 0x5c17ef6cu,
                  0x5e3b0968u) == e);

    Field::sub(
        e,
        Field(0x2a1a37d0u, 0xffb553c4u, 0x22ff7e4au, 0xe256deb5u, 0x18faee65u),
        Field(0x1cfaec75u, 0x7faf7c19u, 0xd3121b9eu, 0xded3ca3bu, 0x952e1b38u));
    REQUIRE(Field(0x0d1f4b5bu, 0x8005d7aau, 0x4fed62acu, 0x03831479u,
                  0x83ccd32du) == e);

    Field::sub(
        e,
        Field(0x223ed22eu, 0x257c787eu, 0xd3505455u, 0xdbded35cu, 0x964eda39u),
        Field(0x4a617461u, 0x4c8165c6u, 0xf378a372u, 0x8d6cccb6u, 0xd07f7850u));
    REQUIRE(Field(0x8f566078u, 0xb1d6a8dfu, 0xd5af7fadu, 0xaa89f612u,
                  0x240a6b52u) == e);
}

TEST_CASE("montgomery mul", "[ring][field]") {
    const Field &Mod = reinterpret_cast<const Field &>(MOD_160);
    const Field &RR = reinterpret_cast<const Field &>(RR_160);
    const Field One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field a, b;

    Field::mul(a, Field(), RR);
    REQUIRE(Field(0) == a);

    Field::mul(b, a, One);
    REQUIRE(Field(0) == b);

    Field::mul(a, Field(0xffffffffu), RR);
    REQUIRE(Field(0xad37b410u, 0x255c6eb2u, 0x7601a883u, 0x659883e8u,
                  0x070707fcu) == a);

    Field::mul(b, a, One);
    REQUIRE(Field(0xffffffffu) == b);

    Field c, d, e;
    do {
        for (int i = 0; i < LN_160; ++i) {
            c.arr[i] = dis_u32(gen);
        }
    } while (c >= Mod);

    d = c;
    Field::mul(e, d, RR);
    Field::mul(d, e, One);
    REQUIRE(c == d);

    LIMB_T l, h, x, y;
    Field mon_x, mon_y, mon_xy, xy;

    x = 0xd8b2f21eu;
    y = 0xabf7c642u;
    utils::uint_mul_lh(l, h, x, y);
    Field::mul(mon_x, Field(x), RR);
    Field::mul(mon_y, Field(y), RR);
    Field::mul(mon_xy, mon_x, mon_y);
    Field::mul(xy, mon_xy, One);
    REQUIRE(l == xy.get_arr()[0]);
    REQUIRE(h == xy.get_arr()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    Field::mul(mon_x, Field(x), RR);
    Field::mul(mon_y, Field(y), RR);
    Field::mul(mon_xy, mon_x, mon_y);
    Field::mul(xy, mon_xy, One);
    REQUIRE(l == xy.get_arr()[0]);
    REQUIRE(h == xy.get_arr()[1]);

    x = dis_u32(gen);
    y = dis_u32(gen);
    utils::uint_mul_lh(l, h, x, y);
    Field::mul(mon_x, RR, Field(x));
    Field::mul(mon_y, RR, Field(y));
    Field::mul(mon_xy, mon_x, mon_y);
    Field::mul(xy, One, mon_xy);
    REQUIRE(l == xy.get_arr()[0]);
    REQUIRE(h == xy.get_arr()[1]);

    mon_x =
        Field(0xa5481e14u, 0x293b3c7du, 0xb85ecae1u, 0x83d79492u, 0xcd652763u);
    mon_y =
        Field(0x93d20f51u, 0x898541bbu, 0x74aa1184u, 0xbccb10b2u, 0x47f79c2cu);
    Field::mul(mon_xy, mon_x, mon_y);
    REQUIRE(Field(0x4886fd54u, 0x272469d8u, 0x0a283135u, 0xa3e81093u,
                  0xa1c4f697u) == mon_xy);
}

TEST_CASE("montgomery inv", "[field]") {
    const auto &RR = reinterpret_cast<const Field &>(RR_160);
    const Field One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    Field a, mon_a, inv_a, mon_prod, prod, r, s, t;
    for (int k = 0; k < 10000; ++k) {
        do {
            for (int k = 0; k < LN_160; ++k) {
                a.get_arr()[k] = dis_u32(gen);
            }
        } while (a >= RR);
        // a = Field(0x31a50ad6u, 0x93f524b7u, 0xa6ea2efeu, 0xed31237au,
        //           0x2d2731f7u);
        a.println();
        Field::mul(mon_a, a, RR);
        mon_a.println();
        Field::inv(inv_a, mon_a, r, s, t);
        inv_a.println();
        Field::mul(mon_prod, mon_a, inv_a);
        mon_prod.println();
        Field::mul(prod, mon_prod, One);
        prod.println();
        REQUIRE(One == prod);
    }
}

TEST_CASE("montgomery mul bench", "[ring][field][bench]") {
    const auto &MOD = reinterpret_cast<const Field2 &>(MOD2_160);
    const auto &RR = reinterpret_cast<const Field2 &>(RR2_160);
    const Field2 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());
    std::uniform_int_distribution<LIMB2_T> dis_u64(
        std::numeric_limits<LIMB2_T>::min(),
        std::numeric_limits<LIMB2_T>::max());
    Field2 x0, y0, mon_x0, mon_y0;
    do {
        x0.get_arr()[0] = dis_u64(gen);
        x0.get_arr()[1] = dis_u64(gen);
        x0.get_arr()[2] = dis_u32(gen);
    } while (x0 >= MOD);
    do {
        y0.get_arr()[0] = dis_u64(gen);
        y0.get_arr()[1] = dis_u64(gen);
        y0.get_arr()[2] = dis_u32(gen);
    } while (y0 >= MOD);
    Field2::mul(mon_x0, x0, RR);
    Field2::mul(mon_y0, y0, RR);

    {
        using F = Field;
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
        using F = Field2;
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
    const auto &MOD = reinterpret_cast<const Field2 &>(MOD2_160);
    const auto &RR = reinterpret_cast<const Field2 &>(RR2_160);
    const Field2 One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());
    std::uniform_int_distribution<LIMB2_T> dis_u64(
        std::numeric_limits<LIMB2_T>::min(),
        std::numeric_limits<LIMB2_T>::max());
    Field2 x0, mon_x0, r0, s0, t0;
    do {
        x0.get_arr()[0] = dis_u64(gen);
        x0.get_arr()[1] = dis_u64(gen);
        x0.get_arr()[2] = dis_u32(gen);
    } while (x0 >= MOD);
    Field2::mul(mon_x0, x0, RR);

    {
        using F = Field;
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        F &r = reinterpret_cast<F &>(r0);
        F &s = reinterpret_cast<F &>(s0);
        F &t = reinterpret_cast<F &>(t0);
        BENCHMARK("32-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x, r, s, t);
            return inv_x;
        };
    }

    {
        using F = Field2;
        const F &mon_x = reinterpret_cast<const F &>(mon_x0);
        F &r = reinterpret_cast<F &>(r0);
        F &s = reinterpret_cast<F &>(s0);
        F &t = reinterpret_cast<F &>(t0);
        BENCHMARK("64-bits montgomery inv") {
            F inv_x;
            F::inv(inv_x, mon_x, r, s, t);
            return inv_x;
        };
    }
}
