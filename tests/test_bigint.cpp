#include "common.hpp"

#include <gec/bigint.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

class AddG160 : public ArrayBE<LIMB_T, LN_160>,
                public Constants<AddG160, LIMB_T, LN_160>,
                public VtCompare<AddG160, LIMB_T, LN_160>,
                public BitOps<AddG160, LIMB_T, LN_160>,
                public AddSubMixin<AddG160, LIMB_T, LN_160>,
                public ArrayOstream<AddG160, LIMB_T, LN_160> {
  public:
    using ArrayBE::ArrayBE;
};

TEST_CASE("bigint constructor", "[bigint]") {
    AddG160 e0;
    REQUIRE(e0.array()[0] == 0);
    REQUIRE(e0.array()[1] == 0);
    REQUIRE(e0.array()[2] == 0);
    REQUIRE(e0.array()[3] == 0);
    REQUIRE(e0.array()[4] == 0);

    AddG160 e1(0x1234);
    REQUIRE(e1.array()[0] == 0x1234);
    REQUIRE(e1.array()[1] == 0);
    REQUIRE(e1.array()[2] == 0);
    REQUIRE(e1.array()[3] == 0);
    REQUIRE(e1.array()[4] == 0);

    AddG160 e2(1, 2, 3, 4, 5);
    REQUIRE(e2.array()[0] == 5);
    REQUIRE(e2.array()[1] == 4);
    REQUIRE(e2.array()[2] == 3);
    REQUIRE(e2.array()[3] == 2);
    REQUIRE(e2.array()[4] == 1);

    AddG160 e3(e2);
    REQUIRE(e3 == e2);
    REQUIRE(e3 != e0);
}

TEST_CASE("bigint comparison", "[bigint]") {
    AddG160 e0, e1(0x0), e2(0x1), e3(0x0, 0x0, 0x0, 0x1, 0x0),
        e4(0x0, 0x0, 0x0, 0x1, 0x1), e5(0x1, 0x0, 0x0, 0x0, 0x0),
        e6(0x1, 0x0, 0x1, 0x0, 0x0);

    REQUIRE(e0 == e1);
    REQUIRE(!(e1 == e2));
    REQUIRE(!(e2 == e3));
    REQUIRE(!(e3 == e4));
    REQUIRE(!(e4 == e5));
    REQUIRE(!(e5 == e6));

    REQUIRE(!(e0 != e1));
    REQUIRE(e1 != e2);
    REQUIRE(e2 != e3);
    REQUIRE(e3 != e4);
    REQUIRE(e4 != e5);
    REQUIRE(e5 != e6);

    REQUIRE(!(e0 < e1));
    REQUIRE(e1 < e2);
    REQUIRE(e2 < e3);
    REQUIRE(e3 < e4);
    REQUIRE(e4 < e5);
    REQUIRE(e5 < e6);

    REQUIRE(e0 <= e1);
    REQUIRE(e1 <= e2);
    REQUIRE(e2 <= e3);
    REQUIRE(e3 <= e4);
    REQUIRE(e4 <= e5);
    REQUIRE(e5 <= e6);

    REQUIRE(!(e0 > e1));
    REQUIRE(!(e1 > e2));
    REQUIRE(!(e2 > e3));
    REQUIRE(!(e3 > e4));
    REQUIRE(!(e4 > e5));
    REQUIRE(!(e5 > e6));

    REQUIRE(e0 >= e1);
    REQUIRE(!(e1 >= e2));
    REQUIRE(!(e2 >= e3));
    REQUIRE(!(e3 >= e4));
    REQUIRE(!(e4 >= e5));
    REQUIRE(!(e5 >= e6));
}

TEST_CASE("bigint bit operations", "[bigint]") {
    using F = AddG160;
    F a(0x0ffff000u, 0x0000ffffu, 0xffffffffu, 0xffffffffu, 0x00000000u);
    F b(0x000ffff0u, 0xffff0000u, 0x00000000u, 0xffffffffu, 0x00000000u);
    F c;

    c.bit_and(a, b);
    REQUIRE(F(0x000ff000u, 0x00000000u, 0x00000000u, 0xffffffffu,
              0x00000000u) == c);
    c.bit_or(a, b);
    REQUIRE(F(0x0ffffff0u, 0xffffffffu, 0xffffffffu, 0xffffffffu,
              0x00000000u) == c);
    c.bit_not(a);
    REQUIRE(F(0xf0000fffu, 0xffff0000u, 0x00000000u, 0x00000000u,
              0xffffffffu) == c);
    c.bit_xor(a, b);
    REQUIRE(F(0x0ff00ff0u, 0xffffffffu, 0xffffffffu, 0x00000000u,
              0x00000000u) == c);

    REQUIRE(156 == a.most_significant_bit());
    REQUIRE(148 == b.most_significant_bit());
    c.set_zero();
    REQUIRE(0 == c.most_significant_bit());
    c.set_one();
    REQUIRE(1 == c.most_significant_bit());
}

TEST_CASE("bigint shift", "[bigint]") {
    AddG160 e(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu, 0xf001000fu);

    e.shift_right<0>();
    REQUIRE(AddG160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                    0xf001000fu) == e);

    e.shift_right<3>();
    REQUIRE(AddG160(0x1e00a001u, 0xfe008001u, 0xfe006001u, 0xfe004001u,
                    0xfe002001u) == e);

    e.shift_right<32>();
    REQUIRE(AddG160(0x00000000u, 0x1e00a001u, 0xfe008001u, 0xfe006001u,
                    0xfe004001u) == e);

    e.shift_right<33>();
    REQUIRE(AddG160(0x00000000u, 0x00000000u, 0x0f005000u, 0xff004000u,
                    0xff003000u) == e);

    e.shift_right<66>();
    REQUIRE(AddG160(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
                    0x03c01400u) == e);

    e.shift_right<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_right<32 * 5 + 1>(); // don't do that

    e = AddG160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                0xf001000fu);

    e.shift_left<0>();
    REQUIRE(AddG160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                    0xf001000fu) == e);

    e.shift_left<3>();
    REQUIRE(AddG160(0x8028007fu, 0x8020007fu, 0x8018007fu, 0x8010007fu,
                    0x80080078u) == e);

    e.shift_left<32>();
    REQUIRE(AddG160(0x8020007fu, 0x8018007fu, 0x8010007fu, 0x80080078u,
                    0x00000000u) == e);

    e.shift_left<33>();
    REQUIRE(AddG160(0x003000ffu, 0x002000ffu, 0x001000f0u, 0x000000000u,
                    0x00000000u) == e);
    // 003000ff 002000ff 001000f0 00000000 00000000
    e.shift_left<66>();
    REQUIRE(AddG160(0x004003c0u, 0x00000000u, 0x00000000u, 0x00000000u,
                    0x00000000) == e);

    e.shift_left<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_left<32 * 5 + 1>(); // don't do that
}

TEST_CASE("bigint add", "[bigint]") {
    AddG160 e;
    bool carry;

    carry = AddG160::add(e, AddG160(), AddG160());
    REQUIRE(e.is_zero());
    REQUIRE(!carry);

    carry = AddG160::add(e, AddG160(0x12), AddG160(0xe));
    REQUIRE(AddG160(0x20) == e);
    REQUIRE(!carry);

    carry = AddG160::add(e, AddG160(0xa2000000u), AddG160(0x5f000000u));
    REQUIRE(AddG160(0, 0, 0, 0x1u, 0x01000000u) == e);
    REQUIRE(!carry);

    carry = AddG160::add(e, AddG160(0xa2000000u, 0x5f000000u, 0, 0, 0),
                         AddG160(0x5f000000u, 0xa2000000u, 0, 0, 0));
    REQUIRE(AddG160(0x01000001u, 0x01000000u, 0, 0, 0) == e);
    REQUIRE(carry);

    e = AddG160();
    carry = AddG160::add(e, AddG160());
    REQUIRE(e.is_zero());
    REQUIRE(!carry);

    e = AddG160(0x12);
    carry = AddG160::add(e, AddG160(0xe));
    REQUIRE(AddG160(0x20) == e);
    REQUIRE(!carry);

    e = AddG160(0xa2000000u);
    carry = AddG160::add(e, AddG160(0x5f000000u));
    REQUIRE(AddG160(0, 0, 0, 0x1u, 0x01000000u) == e);
    REQUIRE(!carry);

    e = AddG160(0xa2000000u, 0x5f000000u, 0, 0, 0);
    carry = AddG160::add(e, AddG160(0x5f000000u, 0xa2000000u, 0, 0, 0));
    REQUIRE(AddG160(0x01000001u, 0x01000000u, 0, 0, 0) == e);
    REQUIRE(carry);
}

TEST_CASE("bigint sub", "[bigint]") {
    AddG160 e;
    bool borrow;

    borrow = AddG160::sub(e, AddG160(), AddG160());
    REQUIRE(e.is_zero());
    REQUIRE(!borrow);

    borrow = AddG160::sub(e, AddG160(0xf0), AddG160(0x2));
    REQUIRE(AddG160(0xee) == e);
    REQUIRE(!borrow);

    borrow = AddG160::sub(e, AddG160(0x10000000u, 0, 0, 0, 0), AddG160(0x1));
    REQUIRE(AddG160(0x0fffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                    0xffffffffu) == e);
    REQUIRE(!borrow);

    borrow = AddG160::sub(e, AddG160(0, 0, 0, 0, 0), AddG160(0x1));
    REQUIRE(AddG160(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                    0xffffffffu) == e);
    REQUIRE(borrow);

    borrow = AddG160::sub(e,
                          AddG160(0x96eb8e57u, 0xa17e5730u, 0x336ebe5eu,
                                  0x553bdef2u, 0xfc26eb86u),
                          AddG160(0x438ab2ceu, 0xa07f9675u, 0x30debdd3u,
                                  0xc9446c1bu, 0x85b4ff59u));
    REQUIRE(AddG160(0x5360db89u, 0x00fec0bbu, 0x0290008au, 0x8bf772d7u,
                    0x7671ec2du) == e);
    REQUIRE(!borrow);

    borrow = AddG160::sub(e,
                          AddG160(0x01a8b80cu, 0x425b5530u, 0xc29ce6b1u,
                                  0xebc4a008u, 0x107bb597u),
                          AddG160(0x54e006b4u, 0x731480edu, 0x56e01a41u,
                                  0x2aa50851u, 0x852f86a2u));
    REQUIRE(AddG160(0xacc8b157u, 0xcf46d443u, 0x6bbccc70u, 0xc11f97b6u,
                    0x8b4c2ef5u) == e);
    REQUIRE(borrow);

    e = AddG160();
    borrow = AddG160::sub(e, AddG160());
    REQUIRE(e.is_zero());
    REQUIRE(!borrow);

    e = AddG160(0xf0);
    borrow = AddG160::sub(e, AddG160(0x2));
    REQUIRE(AddG160(0xee) == e);
    REQUIRE(!borrow);

    e = AddG160(0x10000000u, 0, 0, 0, 0);
    borrow = AddG160::sub(e, AddG160(0x1));
    REQUIRE(AddG160(0x0fffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                    0xffffffffu) == e);
    REQUIRE(!borrow);

    e = AddG160(0, 0, 0, 0, 0);
    borrow = AddG160::sub(e, AddG160(0x1));
    REQUIRE(AddG160(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                    0xffffffffu) == e);
    REQUIRE(borrow);

    e = AddG160(0x96eb8e57u, 0xa17e5730u, 0x336ebe5eu, 0x553bdef2u,
                0xfc26eb86u);
    borrow = AddG160::sub(e, AddG160(0x438ab2ceu, 0xa07f9675u, 0x30debdd3u,
                                     0xc9446c1bu, 0x85b4ff59u));
    REQUIRE(AddG160(0x5360db89u, 0x00fec0bbu, 0x0290008au, 0x8bf772d7u,
                    0x7671ec2du) == e);
    REQUIRE(!borrow);

    e = AddG160(0x01a8b80cu, 0x425b5530u, 0xc29ce6b1u, 0xebc4a008u,
                0x107bb597u);
    borrow = AddG160::sub(e, AddG160(0x54e006b4u, 0x731480edu, 0x56e01a41u,
                                     0x2aa50851u, 0x852f86a2u));
    REQUIRE(AddG160(0xacc8b157u, 0xcf46d443u, 0x6bbccc70u, 0xc11f97b6u,
                    0x8b4c2ef5u) == e);
    REQUIRE(borrow);
}