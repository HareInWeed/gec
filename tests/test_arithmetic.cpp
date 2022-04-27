#include "common.hpp"
#include "utils.hpp"

#include <gec/utils/arithmetic.hpp>

#include <limits>
#include <random>

#include "configured_catch.hpp"

using namespace gec;

TEST_CASE("uint_add_with_carry", "[arithmetic]") {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<uint32_t> dis_u32(
        std::numeric_limits<uint32_t>::min(),
        std::numeric_limits<uint32_t>::max());
    const uint32_t a = dis_u32(gen);
    const uint32_t b = dis_u32(gen);
    uint32_t c;
    bool carry;

    uint64_t result;

    c = a;
    carry = utils::uint_add_with_carry(c, b, false);
    result = uint64_t(a) + uint64_t(b);
    REQUIRE(uint32_t(result) == c);
    REQUIRE(carry == ((result >> 32) != 0));

    c = a;
    carry = utils::uint_add_with_carry(c, b, true);
    result = uint64_t(a) + uint64_t(b) + 1;
    REQUIRE(uint32_t(result) == c);
    REQUIRE(carry == ((result >> 32) != 0));
}

TEST_CASE("uint_mul_lh", "[arithmetic]") {
    uint32_t l, h;

    utils::uint_mul_lh(l, h, 0u, 0u);
    REQUIRE(0 == h);
    REQUIRE(0 == l);

    utils::uint_mul_lh(l, h, 0xffffffffu, 2u);
    REQUIRE(1 == h);
    REQUIRE(0xfffffffeu == l);

    utils::uint_mul_lh(l, h, 0xffffffffu, 0xffffffffu);
    REQUIRE(0xfffffffeu == h);
    REQUIRE(0x00000001u == l);

    utils::uint_mul_lh(l, h, 0xffffffffu, 0xffffffffu);
    REQUIRE(0xfffffffeu == h);
    REQUIRE(0x00000001u == l);

    utils::uint_mul_lh(l, h, 0xed04d507u, 0x72d0f643u);
    REQUIRE(0x6a4d9ecau == h);
    REQUIRE(0xe0a87ad5u == l);

    using Num = OpaqueNum<uint32_t>;
    Num lo, ho;

    utils::uint_mul_lh<Num>(lo, ho, 0u, 0u);
    REQUIRE(Num(0) == ho);
    REQUIRE(Num(0) == lo);

    utils::uint_mul_lh<Num>(lo, ho, 0xffffffffu, 2u);
    REQUIRE(Num(1) == ho);
    REQUIRE(Num(0xfffffffeu) == lo);

    utils::uint_mul_lh<Num>(lo, ho, 0xffffffffu, 0xffffffffu);
    REQUIRE(Num(0xfffffffeu) == ho);
    REQUIRE(Num(0x00000001u) == lo);

    utils::uint_mul_lh<Num>(lo, ho, 0xffffffffu, 0xffffffffu);
    REQUIRE(Num(0xfffffffeu) == ho);
    REQUIRE(Num(0x00000001u) == lo);

    utils::uint_mul_lh<Num>(lo, ho, 0xed04d507u, 0x72d0f643u);
    REQUIRE(Num(0x6a4d9ecau) == ho);
    REQUIRE(Num(0xe0a87ad5u) == lo);

    {
        uint64_t l, h;
        utils::uint_mul_lh(l, h, 0xdf53a07139176865, 0xb8399c8e2ebb2d0du);
        REQUIRE(0xa0b65d83192ad70eu == h);
        REQUIRE(0x5a8069aa6b510e21u == l);
    }

    {
        using Num = OpaqueNum<uint64_t>;
        Num l, h;
        utils::uint_mul_lh<Num>(l, h, 0xdf53a07139176865, 0xb8399c8e2ebb2d0du);
        REQUIRE(Num(0xa0b65d83192ad70eu) == h);
        REQUIRE(Num(0x5a8069aa6b510e21u) == l);
    }
}

TEST_CASE("uint_mul_lh bench", "[arithmetic][bench]") {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<uint32_t> dis_u32(
        std::numeric_limits<uint32_t>::min(),
        std::numeric_limits<uint32_t>::max());
    uint32_t a_u32 = dis_u32(gen);
    uint32_t b_u32 = dis_u32(gen);

    std::uniform_int_distribution<uint64_t> dis_u64(
        std::numeric_limits<uint64_t>::min(),
        std::numeric_limits<uint64_t>::max());
    uint64_t a_u64 = dis_u64(gen);
    uint64_t b_u64 = dis_u64(gen);

    BENCHMARK("uint_mul_lh uint32_t specialized") {
        uint32_t l, h;
        utils::uint_mul_lh(l, h, a_u32, b_u32);
        return h;
    };

    BENCHMARK("uint_mul_lh uint32_t generic") {
        using Num = OpaqueNum<uint32_t>;
        Num l, h;
        utils::uint_mul_lh<Num>(l, h, a_u32, b_u32);
        return h;
    };

    BENCHMARK("uint_mul_lh uint64_t specialized") {
        uint64_t l, h;
        utils::uint_mul_lh(l, h, a_u64, b_u64);
        return h;
    };

    BENCHMARK("uint_mul_lh uint64_t generic") {
        using Num = OpaqueNum<uint64_t>;
        Num l, h;
        utils::uint_mul_lh<Num>(l, h, a_u32, b_u64);
        return h;
    };
}

TEST_CASE("seq_add_mul_limb", "[arithmetic]") {
    uint32_t a[LN_160] = {0xde15b1e0u, 0x7788c10cu, 0xc66edfbfu, 0x0bc7a6f0u,
                          0x113586c5u};
    const uint32_t b[LN_160] = {0x5e3b0969u, 0x5c17ef6cu, 0xf5d7cecau,
                                0xd8db9627u, 0xb77902abu};
    const uint32_t x = 2098395424;
    const uint32_t last_limb = utils::seq_add_mul_limb<LN_160>(a, MOD160, x);
    REQUIRE(a[0] == 0x00000000u);
    REQUIRE(a[1] == 0x020889ffu);
    REQUIRE(a[2] == 0xc2bdb635u);
    REQUIRE(a[3] == 0xce20f333u);
    REQUIRE(a[4] == 0xcce779efu);
    REQUIRE(last_limb == 0x59a3af5bu);
}