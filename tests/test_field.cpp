#include "test_common.hpp"

#include <gec/bigint.hpp>
#include <gec/bigint/mixin/ostream.hpp>

#include <catch2/catch.hpp>

using namespace gec;
using namespace bigint;

class Field : public Array<LIMB_T, LIMB_N>,
              public VtCompareMixin<Field, LIMB_T, LIMB_N>,
              public BitOpsMixin<Field, LIMB_T, LIMB_N>,
              public ModAddSubMixin<Field, LIMB_T, LIMB_N, MOD>,
              public ArrayOstreamMixin<Field, LIMB_T, LIMB_N> {
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
