#include "common.hpp"

#include <gec/bigint.hpp>
#include <gec/bigint/mixin/ostream.hpp>
#include <gec/bigint/mixin/print.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

class Field : public Array<LIMB_T, LN_160>,
              public VtCompareMixin<Field, LIMB_T, LN_160>,
              public BitOpsMixin<Field, LIMB_T, LN_160>,
              public ModAddSubMixin<Field, LIMB_T, LN_160, MOD_160>,
              public Montgomery<Field, LIMB_T, LN_160, MOD_160, MOD_P_160>,
              public ArrayOstreamMixin<Field, LIMB_T, LN_160>,
              public ArrayPrintMixin<Field, LIMB_T, LN_160> {
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

TEST_CASE("montgomery", "[ring][field]") {
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
}

TEST_CASE("montgomery bench", "[ring][field][bench]") {
    const Field &RR = reinterpret_cast<const Field &>(RR_160);
    const Field One(1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());
    Field x, y, mon_x, mon_y;
    do {
        for (int i = 0; i < LN_160; ++i) {
            x.arr[i] = dis_u32(gen);
        }
    } while (x < RR);
    do {
        for (int i = 0; i < LN_160; ++i) {
            y.arr[i] = dis_u32(gen);
        }
    } while (x < RR);
    Field::mul(mon_x, x, RR);
    Field::mul(mon_y, y, RR);

    BENCHMARK("into montgomery form") {
        Field res;
        Field::mul(res, x, RR);
        return res;
    };

    BENCHMARK("from montgomery form") {
        Field res;
        Field::mul(res, mon_x, One);
        return res;
    };

    BENCHMARK("montgomery mul") {
        Field xy;
        Field::mul(xy, x, y);
        return xy;
    };
}