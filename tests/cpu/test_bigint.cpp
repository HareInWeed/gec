#include <common.hpp>

#include <gec/bigint.hpp>

#include <configured_catch.hpp>

#ifdef GEC_NVCC
GEC_INT_TOO_LARGE
#endif // GEC_NVCC

using namespace gec;
using namespace bigint;
using namespace gec::bigint::literal;

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
    Bigint160 e(0xf005000f'f004000f'f003000f'f002000f'f001000f_int);

    e.shift_right<0>();
    REQUIRE(Bigint160(0xf005000f'f004000f'f003000f'f002000f'f001000f_int) == e);

    e.shift_right<3>();
    REQUIRE(Bigint160(0x1e00a001'fe008001'fe006001'fe004001'fe002001_int) == e);

    e.shift_right<32>();
    REQUIRE(Bigint160(0x00000000'1e00a001'fe008001'fe006001'fe004001_int) == e);

    e.shift_right<33>();
    REQUIRE(Bigint160(0x00000000'00000000'0f005000'ff004000'ff003000_int) == e);

    e.shift_right<66>();
    REQUIRE(Bigint160(0x00000000'00000000'00000000'00000000'03c01400_int) == e);

    e.shift_right<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_right<32 * 5 + 1>(); // don't do that

    e = Bigint160(0xf005000f'f004000f'f003000f'f002000f'f001000f_int);

    e.shift_left<0>();
    REQUIRE(Bigint160(0xf005000f'f004000f'f003000f'f002000f'f001000f_int) == e);

    e.shift_left<3>();
    REQUIRE(Bigint160(0x8028007f'8020007f'8018007f'8010007f'80080078_int) == e);

    e.shift_left<32>();
    REQUIRE(Bigint160(0x8020007f'8018007f'8010007f'80080078'00000000_int) == e);

    e.shift_left<33>();
    REQUIRE(Bigint160(0x003000ff'002000ff'001000f0'00000000'00000000_int) == e);
    // 003000ff 002000ff 001000f0 00000000 00000000
    e.shift_left<66>();
    REQUIRE(Bigint160(0x004003c0'00000000'00000000'00000000'00000000_int) == e);

    e.shift_left<32 * 5>();
    REQUIRE(e.is_zero());

    // e.shift_left<32 * 5 + 1>(); // don't do that
}

TEST_CASE("bigint bit operations", "[bigint]") {
    using F = Bigint160;
    F a(0x0ffff000'0000ffff'ffffffff'ffffffff'00000000_int);
    F b(0x000ffff0'ffff0000'00000000'ffff0000'00000000_int);
    F c;

    c.bit_and(a, b);
    REQUIRE(F(0x000ff000'00000000'00000000'ffff0000'00000000_int) == c);
    c.bit_or(a, b);
    REQUIRE(F(0x0ffffff0'ffffffff'ffffffff'ffffffff'00000000_int) == c);
    c.bit_not(a);
    REQUIRE(F(0xf0000fff'ffff0000'00000000'00000000'ffffffff_int) == c);
    c.bit_xor(a, b);
    REQUIRE(F(0x0ff00ff0'ffffffff'ffffffff'0000ffff'00000000_int) == c);

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

    carry =
        Bigint160::add(e, Bigint160(0xa2000000_int), Bigint160(0x5f000000_int));
    REQUIRE(Bigint160(0x1'01000000_int) == e);
    REQUIRE(!carry);

    carry = Bigint160::add(
        e, Bigint160(0xa2000000'5f000000'00000000'00000000'00000000_int),
        Bigint160(0x5f000000'a2000000'00000000'00000000'00000000_int));
    REQUIRE(Bigint160(0x01000001'01000000'00000000'00000000'00000000_int) == e);
    REQUIRE(carry);

    e = Bigint160();
    carry = Bigint160::add(e, Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!carry);

    e = Bigint160(0x12);
    carry = Bigint160::add(e, Bigint160(0xe));
    REQUIRE(Bigint160(0x20) == e);
    REQUIRE(!carry);

    e = Bigint160(0xa2000000_int);
    carry = Bigint160::add(e, Bigint160(0x5f000000_int));
    REQUIRE(Bigint160(0x1'01000000_int) == e);
    REQUIRE(!carry);

    e = Bigint160(0xa2000000'5f000000'00000000'00000000'00000000_int);
    carry = Bigint160::add(
        e, Bigint160(0x5f000000'a2000000'00000000'00000000'00000000_int));
    REQUIRE(Bigint160(0x01000001'01000000'00000000'00000000'00000000_int) == e);
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

    borrow = Bigint160::sub(
        e, Bigint160(0x10000000'00000000'00000000'00000000'00000000_int),
        Bigint160(0x1));
    REQUIRE(Bigint160(0x0fffffff'ffffffff'ffffffff'ffffffff'ffffffff_int) == e);
    REQUIRE(!borrow);

    borrow = Bigint160::sub(e, Bigint160(0, 0, 0, 0, 0), Bigint160(0x1));
    REQUIRE(Bigint160(0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff_int) == e);
    REQUIRE(borrow);

    borrow = Bigint160::sub(
        e, Bigint160(0x96eb8e57'a17e5730'336ebe5e'553bdef2'fc26eb86_int),
        Bigint160(0x438ab2ce'a07f9675'30debdd3'c9446c1b'85b4ff59_int));
    REQUIRE(Bigint160(0x5360db89'00fec0bb'0290008a'8bf772d7'7671ec2d_int) == e);
    REQUIRE(!borrow);

    borrow = Bigint160::sub(
        e, Bigint160(0x01a8b80c'425b5530'c29ce6b1'ebc4a008'107bb597_int),
        Bigint160(0x54e006b4'731480ed'56e01a41'2aa50851'852f86a2_int));
    REQUIRE(Bigint160(0xacc8b157'cf46d443'6bbccc70'c11f97b6'8b4c2ef5_int) == e);
    REQUIRE(borrow);

    e = Bigint160();
    borrow = Bigint160::sub(e, Bigint160());
    REQUIRE(e.is_zero());
    REQUIRE(!borrow);

    e = Bigint160(0xf0);
    borrow = Bigint160::sub(e, Bigint160(0x2));
    REQUIRE(Bigint160(0xee) == e);
    REQUIRE(!borrow);

    e = Bigint160(0x10000000'00000000'00000000'00000000'00000000_int);
    borrow = Bigint160::sub(e, Bigint160(0x1));
    REQUIRE(Bigint160(0x0fffffff'ffffffff'ffffffff'ffffffff'ffffffff_int) == e);
    REQUIRE(!borrow);

    e = Bigint160(0, 0, 0, 0, 0);
    borrow = Bigint160::sub(e, Bigint160(0x1));
    REQUIRE(Bigint160(0xffffffff'ffffffff'ffffffff'ffffffff'ffffffff_int) == e);
    REQUIRE(borrow);

    e = Bigint160(0x96eb8e57'a17e5730'336ebe5e'553bdef2'fc26eb86_int);
    borrow = Bigint160::sub(
        e, Bigint160(0x438ab2ce'a07f9675'30debdd3'c9446c1b'85b4ff59_int));
    REQUIRE(Bigint160(0x5360db89'00fec0bb'0290008a'8bf772d7'7671ec2d_int) == e);
    REQUIRE(!borrow);

    e = Bigint160(0x01a8b80c'425b5530'c29ce6b1'ebc4a008'107bb597_int);
    borrow = Bigint160::sub(
        e, Bigint160(0x54e006b4'731480ed'56e01a41'2aa50851'852f86a2_int));
    REQUIRE(Bigint160(0xacc8b157'cf46d443'6bbccc70'c11f97b6'8b4c2ef5_int) == e);
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

    T prod[2 * N];

    Int a, b, q, r, q1, r1;
    Int lower(1), upper;
    for (size_t j = 0; j < 2 * N; ++j) {
        upper.array()[j / 2] =
            j & 1 ? LowerKMask<T, bits>::value : LowerKMask<T, bits / 2>::value;
        CAPTURE(upper);
        for (int k = 0; k < 1000; ++k) {
            Int::sample(a, rng);
            Int::sample(b, lower, upper, rng);
            CAPTURE(a, b);

            Int::div_rem(q, r, a, b);
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

            Int::div(q1, a, b);
            REQUIRE(q == q1);
            Int::rem(r1, a, b);
            REQUIRE(r == r1);
        }
    }
}

TEST_CASE("bigint division", "[bigint]") {
    std::random_device rd;
    test_division<Bigint160>(rd());
    test_division<Bigint160_2>(rd());
}