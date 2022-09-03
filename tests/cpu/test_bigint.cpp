#include <common.hpp>

#include <gec/bigint.hpp>

#include <configured_catch.hpp>

using namespace gec;
using namespace bigint;

using Bigint160 = Bigint<LIMB_T, LN_160>;
using Bigint160_2 = Bigint<LIMB2_T, LN2_160>;

TEST_CASE("bigint constructor", "[bigint]") {
    Bigint160 e0;
    REQUIRE(e0.array()[0] == 0);
    REQUIRE(e0.array()[1] == 0);
    REQUIRE(e0.array()[2] == 0);
    REQUIRE(e0.array()[3] == 0);
    REQUIRE(e0.array()[4] == 0);

    Bigint160 e1(0x1234);
    REQUIRE(e1.array()[0] == 0x1234);
    REQUIRE(e1.array()[1] == 0);
    REQUIRE(e1.array()[2] == 0);
    REQUIRE(e1.array()[3] == 0);
    REQUIRE(e1.array()[4] == 0);

    Bigint160 e2(1, 2, 3, 4, 5);
    REQUIRE(e2.array()[0] == 5);
    REQUIRE(e2.array()[1] == 4);
    REQUIRE(e2.array()[2] == 3);
    REQUIRE(e2.array()[3] == 2);
    REQUIRE(e2.array()[4] == 1);

    Bigint160 e3(e2);
    REQUIRE(e3 == e2);
    REQUIRE(e3 != e0);
}

TEST_CASE("bigint comparison", "[bigint]") {
    Bigint160 e0, e1(0x0), e2(0x1), e3(0x0, 0x0, 0x0, 0x1, 0x0),
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

TEST_CASE("bigint shift", "[bigint]") {
    Bigint160 e(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                0xf001000fu);

    e.shift_right<0>();
    REQUIRE(Bigint160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                      0xf001000fu) == e);

    e.shift_right<3>();
    REQUIRE(Bigint160(0x1e00a001u, 0xfe008001u, 0xfe006001u, 0xfe004001u,
                      0xfe002001u) == e);

    e.shift_right<32>();
    REQUIRE(Bigint160(0x00000000u, 0x1e00a001u, 0xfe008001u, 0xfe006001u,
                      0xfe004001u) == e);

    e.shift_right<33>();
    REQUIRE(Bigint160(0x00000000u, 0x00000000u, 0x0f005000u, 0xff004000u,
                      0xff003000u) == e);

    e.shift_right<66>();
    REQUIRE(Bigint160(0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
                      0x03c01400u) == e);

    e.shift_right<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_right<32 * 5 + 1>(); // don't do that

    e = Bigint160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                  0xf001000fu);

    e.shift_left<0>();
    REQUIRE(Bigint160(0xf005000fu, 0xf004000fu, 0xf003000fu, 0xf002000fu,
                      0xf001000fu) == e);

    e.shift_left<3>();
    REQUIRE(Bigint160(0x8028007fu, 0x8020007fu, 0x8018007fu, 0x8010007fu,
                      0x80080078u) == e);

    e.shift_left<32>();
    REQUIRE(Bigint160(0x8020007fu, 0x8018007fu, 0x8010007fu, 0x80080078u,
                      0x00000000u) == e);

    e.shift_left<33>();
    REQUIRE(Bigint160(0x003000ffu, 0x002000ffu, 0x001000f0u, 0x000000000u,
                      0x00000000u) == e);
    // 003000ff 002000ff 001000f0 00000000 00000000
    e.shift_left<66>();
    REQUIRE(Bigint160(0x004003c0u, 0x00000000u, 0x00000000u, 0x00000000u,
                      0x00000000) == e);

    e.shift_left<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_left<32 * 5 + 1>(); // don't do that
}

TEST_CASE("bigint bit operations", "[bigint]") {
    using F = Bigint160;
    F a(0x0ffff000u, 0x0000ffffu, 0xffffffffu, 0xffffffffu, 0x00000000u);
    F b(0x000ffff0u, 0xffff0000u, 0x00000000u, 0xffff0000u, 0x00000000u);
    F c;

    c.bit_and(a, b);
    REQUIRE(F(0x000ff000u, 0x00000000u, 0x00000000u, 0xffff0000u,
              0x00000000u) == c);
    c.bit_or(a, b);
    REQUIRE(F(0x0ffffff0u, 0xffffffffu, 0xffffffffu, 0xffffffffu,
              0x00000000u) == c);
    c.bit_not(a);
    REQUIRE(F(0xf0000fffu, 0xffff0000u, 0x00000000u, 0x00000000u,
              0xffffffffu) == c);
    c.bit_xor(a, b);
    REQUIRE(F(0x0ff00ff0u, 0xffffffffu, 0xffffffffu, 0x0000ffffu,
              0x00000000u) == c);

    REQUIRE(155 == a.most_significant_bit());
    REQUIRE(4 == a.leading_zeros());
    REQUIRE(32 == a.least_significant_bit());
    REQUIRE(32 == a.trailing_zeros());

    REQUIRE(147 == b.most_significant_bit());
    REQUIRE(12 == b.leading_zeros());
    REQUIRE(48 == b.least_significant_bit());
    REQUIRE(48 == b.trailing_zeros());

    c.set_zero();
    REQUIRE(160 == c.most_significant_bit());
    REQUIRE(160 == c.leading_zeros());
    REQUIRE(160 == c.least_significant_bit());
    REQUIRE(160 == c.trailing_zeros());
    c.set_one();
    REQUIRE(0 == c.most_significant_bit());
    REQUIRE(159 == c.leading_zeros());
    REQUIRE(0 == c.least_significant_bit());
    REQUIRE(0 == c.trailing_zeros());
}

TEST_CASE("bigint runtime shift", "[bigint]") {
    std::random_device rd;
    auto seed = rd();
    CAPTURE(seed);

    auto rng = make_gec_rng(std::mt19937(seed));

    Bigint160 x, res, expected;
    Bigint160::sample(x, rng);

#define test_helper(bit)                                                       \
    do {                                                                       \
        INFO("shift left: " << bit);                                           \
        expected = x;                                                          \
        expected.shift_left<(bit)>();                                          \
        res = x;                                                               \
        res.shift_left((bit));                                                 \
        REQUIRE(expected == res);                                              \
        INFO("shift right: " << bit);                                          \
        expected = x;                                                          \
        expected.shift_right<(bit)>();                                         \
        res = x;                                                               \
        res.shift_right((bit));                                                \
        REQUIRE(expected == res);                                              \
    } while (0)

    for (int k = 0; k < 10000; ++k) {
        test_helper(0);
        test_helper(1);
        test_helper(7);
        test_helper(31);
        test_helper(32);
        test_helper(33);
        test_helper(64);
        test_helper(65);
        test_helper(66);
        test_helper(32 * 4 - 3);
        test_helper(32 * 4);
        test_helper(32 * 4 + 7);
        test_helper(32 * 5 - 1);
        test_helper(32 * 5);
    }

#undef test_helper
}

TEST_CASE("bigint add", "[bigint]") {
    Bigint160 e;
    bool carry;

    carry = Bigint160::add(e, Bigint160(), Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!carry);

    carry = Bigint160::add(e, Bigint160(0x12), Bigint160(0xe));
    REQUIRE(Bigint160(0x20) == e);
    REQUIRE(!carry);

    carry = Bigint160::add(e, Bigint160(0xa2000000u), Bigint160(0x5f000000u));
    REQUIRE(Bigint160(0, 0, 0, 0x1u, 0x01000000u) == e);
    REQUIRE(!carry);

    carry = Bigint160::add(e, Bigint160(0xa2000000u, 0x5f000000u, 0, 0, 0),
                           Bigint160(0x5f000000u, 0xa2000000u, 0, 0, 0));
    REQUIRE(Bigint160(0x01000001u, 0x01000000u, 0, 0, 0) == e);
    REQUIRE(carry);

    e = Bigint160();
    carry = Bigint160::add(e, Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!carry);

    e = Bigint160(0x12);
    carry = Bigint160::add(e, Bigint160(0xe));
    REQUIRE(Bigint160(0x20) == e);
    REQUIRE(!carry);

    e = Bigint160(0xa2000000u);
    carry = Bigint160::add(e, Bigint160(0x5f000000u));
    REQUIRE(Bigint160(0, 0, 0, 0x1u, 0x01000000u) == e);
    REQUIRE(!carry);

    e = Bigint160(0xa2000000u, 0x5f000000u, 0, 0, 0);
    carry = Bigint160::add(e, Bigint160(0x5f000000u, 0xa2000000u, 0, 0, 0));
    REQUIRE(Bigint160(0x01000001u, 0x01000000u, 0, 0, 0) == e);
    REQUIRE(carry);
}

TEST_CASE("bigint sub", "[bigint]") {
    Bigint160 e;
    bool borrow;

    borrow = Bigint160::sub(e, Bigint160(), Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!borrow);

    borrow = Bigint160::sub(e, Bigint160(0xf0), Bigint160(0x2));
    REQUIRE(Bigint160(0xee) == e);
    REQUIRE(!borrow);

    borrow =
        Bigint160::sub(e, Bigint160(0x10000000u, 0, 0, 0, 0), Bigint160(0x1));
    REQUIRE(Bigint160(0x0fffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                      0xffffffffu) == e);
    REQUIRE(!borrow);

    borrow = Bigint160::sub(e, Bigint160(0, 0, 0, 0, 0), Bigint160(0x1));
    REQUIRE(Bigint160(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                      0xffffffffu) == e);
    REQUIRE(borrow);

    borrow = Bigint160::sub(e,
                            Bigint160(0x96eb8e57u, 0xa17e5730u, 0x336ebe5eu,
                                      0x553bdef2u, 0xfc26eb86u),
                            Bigint160(0x438ab2ceu, 0xa07f9675u, 0x30debdd3u,
                                      0xc9446c1bu, 0x85b4ff59u));
    REQUIRE(Bigint160(0x5360db89u, 0x00fec0bbu, 0x0290008au, 0x8bf772d7u,
                      0x7671ec2du) == e);
    REQUIRE(!borrow);

    borrow = Bigint160::sub(e,
                            Bigint160(0x01a8b80cu, 0x425b5530u, 0xc29ce6b1u,
                                      0xebc4a008u, 0x107bb597u),
                            Bigint160(0x54e006b4u, 0x731480edu, 0x56e01a41u,
                                      0x2aa50851u, 0x852f86a2u));
    REQUIRE(Bigint160(0xacc8b157u, 0xcf46d443u, 0x6bbccc70u, 0xc11f97b6u,
                      0x8b4c2ef5u) == e);
    REQUIRE(borrow);

    e = Bigint160();
    borrow = Bigint160::sub(e, Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!borrow);

    e = Bigint160(0xf0);
    borrow = Bigint160::sub(e, Bigint160(0x2));
    REQUIRE(Bigint160(0xee) == e);
    REQUIRE(!borrow);

    e = Bigint160(0x10000000u, 0, 0, 0, 0);
    borrow = Bigint160::sub(e, Bigint160(0x1));
    REQUIRE(Bigint160(0x0fffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                      0xffffffffu) == e);
    REQUIRE(!borrow);

    e = Bigint160(0, 0, 0, 0, 0);
    borrow = Bigint160::sub(e, Bigint160(0x1));
    REQUIRE(Bigint160(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                      0xffffffffu) == e);
    REQUIRE(borrow);

    e = Bigint160(0x96eb8e57u, 0xa17e5730u, 0x336ebe5eu, 0x553bdef2u,
                  0xfc26eb86u);
    borrow = Bigint160::sub(e, Bigint160(0x438ab2ceu, 0xa07f9675u, 0x30debdd3u,
                                         0xc9446c1bu, 0x85b4ff59u));
    REQUIRE(Bigint160(0x5360db89u, 0x00fec0bbu, 0x0290008au, 0x8bf772d7u,
                      0x7671ec2du) == e);
    REQUIRE(!borrow);

    e = Bigint160(0x01a8b80cu, 0x425b5530u, 0xc29ce6b1u, 0xebc4a008u,
                  0x107bb597u);
    borrow = Bigint160::sub(e, Bigint160(0x54e006b4u, 0x731480edu, 0x56e01a41u,
                                         0x2aa50851u, 0x852f86a2u));
    REQUIRE(Bigint160(0xacc8b157u, 0xcf46d443u, 0x6bbccc70u, 0xc11f97b6u,
                      0x8b4c2ef5u) == e);
    REQUIRE(borrow);
}

TEST_CASE("bigint add limb", "[bigint]") {
    std::random_device rd;
    auto seed = rd();
    CAPTURE(seed);

    using Int = Bigint160;

    std::uniform_int_distribution<Int::LimbT> gen;
    auto rng = make_gec_rng(std::mt19937(seed));

    Int x;
    bool e_carry, carry;
    for (int k = 0; k < 1000; ++k) {
        Int::sample(x, rng);
        Int::LimbT y = gen(rng.get_rng());
        Int yp(y);
        CAPTURE(x, y, yp);

        Int ex, res;
        carry = Int::add(res, x, y);
        e_carry = Int::add(ex, x, yp);
        REQUIRE(ex == res);
        REQUIRE(e_carry == carry);

        ex = x;
        res = x;
        carry = Int::add(res, y);
        e_carry = Int::add(ex, yp);
        REQUIRE(ex == res);
        REQUIRE(e_carry == carry);
    }
}

TEST_CASE("bigint sub limb", "[bigint]") {
    std::random_device rd;
    auto seed = rd();
    CAPTURE(seed);

    using Int = Bigint160;

    std::uniform_int_distribution<Int::LimbT> gen;
    auto rng = make_gec_rng(std::mt19937(seed));

    Int x;
    bool e_borrow, borrow;
    for (int k = 0; k < 1000; ++k) {
        Int::sample(x, rng);
        Int::LimbT y = gen(rng.get_rng());
        Int yp(y);
        CAPTURE(x, y, yp);

        Int ex, res;
        borrow = Int::sub(res, x, y);
        e_borrow = Int::sub(ex, x, yp);
        REQUIRE(ex == res);
        REQUIRE(e_borrow == borrow);

        ex = x;
        res = x;
        borrow = Int::sub(res, y);
        e_borrow = Int::sub(ex, yp);
        REQUIRE(ex == res);
        REQUIRE(e_borrow == borrow);
    }
}

template <typename Int>
static void test_division(std::random_device::result_type seed) {
    using namespace gec::utils;
    using T = typename Int::LimbT;
    constexpr auto bits = type_bits<T>::value;
    constexpr auto N = Int::LimbN;

    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    typename Int::template Context<> ctx;

    T prod[2 * N];

    Int a, b, q, r, q1, r1;
    Int lower(1), upper;
    for (size_t j = 0; j < 2 * N; ++j) {
        upper.array()[j / 2] =
            j & 1 ? LowerKMask<T, bits>::value : LowerKMask<T, bits / 2>::value;
        CAPTURE(upper);
        for (int k = 0; k < 1000; ++k) {
            Int::sample(a, rng);
            Int::sample(b, lower, upper, rng, ctx);
            CAPTURE(a, b);

            Int::div_rem(q, r, a, b, ctx);
            CAPTURE(q, r);

            // product
            fill_seq_limb<2 * N>(prod, T(0));
            T l;
            for (size_t k = 0; k < N; ++k) {
                l = seq_add_mul_limb<N>(prod + k, q.array(), b.array()[k]);
                for (size_t j = N + k; j < 2 * N; ++j) {
                    l = T(uint_add_with_carry(prod[j], l, false));
                }
            }

            // add remainder
            seq_add_limb<N>(prod + N, T(seq_add<N>(prod, r.array())));

            for (size_t k = 0; k < N; ++k) {
                REQUIRE(0 == prod[N + k]);
            }
            for (size_t k = 0; k < N; ++k) {
                REQUIRE(prod[k] == a.array()[k]);
            }

            Int::div(q1, a, b, ctx);
            REQUIRE(q == q1);
            Int::rem(r1, a, b, ctx);
            REQUIRE(r == r1);
        }
    }
}

TEST_CASE("bigint division", "[bigint]") {
    std::random_device rd;
    test_division<Bigint160>(rd());
    test_division<Bigint160_2>(rd());
}