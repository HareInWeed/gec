#include <configured_catch.hpp>

#include <openssl/ec.h>

TEST_CASE("openssl", "[openssl][bench]") {
    BN_CTX *ctx;
    ctx = BN_CTX_new();
    REQUIRE(nullptr != ctx);

    BIGNUM *p = BN_new();
    REQUIRE(nullptr != p);
    REQUIRE(0 != BN_hex2bn(&p, "fffffffffffffffffffffffffffffffffffffffffffffff"
                               "ffffffffefffffc2f"));

    BIGNUM *a = BN_new();
    REQUIRE(nullptr != a);
    REQUIRE(0 != BN_hex2bn(&a, "0"));

    BIGNUM *b = BN_new();
    REQUIRE(nullptr != b);
    REQUIRE(0 != BN_hex2bn(&b, "7"));

    BIGNUM *n = BN_new();
    REQUIRE(nullptr != n);
    REQUIRE(0 != BN_hex2bn(&n, "fffffffffffffffffffffffffffffffebaaedce6af48a03"
                               "bbfd25e8cd0364141"));

    EC_GROUP *ec = EC_GROUP_new_curve_GFp(p, a, b, ctx);
    REQUIRE(nullptr != ec);

    EC_POINT *gen = EC_POINT_new(ec);
    REQUIRE(nullptr != gen);

    BIGNUM *tmp1 = BN_new(), *tmp2 = BN_new();
    REQUIRE(nullptr != tmp1);
    REQUIRE(nullptr != tmp2);

    REQUIRE(0 != BN_hex2bn(&tmp1, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE"
                                  "28D959F2815B16F81798"));
    REQUIRE(0 != BN_hex2bn(&tmp2, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A685"
                                  "54199C47D08FFB10D4B8"));
    REQUIRE(0 != EC_POINT_set_affine_coordinates(ec, gen, tmp1, tmp2, ctx));

    EC_POINT *p1 = EC_POINT_new(ec), *p2 = EC_POINT_new(ec),
             *p3 = EC_POINT_new(ec);
    REQUIRE(nullptr != p1);
    REQUIRE(nullptr != p2);
    REQUIRE(nullptr != p3);

    BENCHMARK_ADVANCED("secp256k1 point add")
    (Catch::Benchmark::Chronometer meter) {
        BN_rand_range(tmp1, n);
        EC_POINT_mul(ec, p1, nullptr, gen, tmp1, ctx);
        BN_rand_range(tmp1, n);
        EC_POINT_mul(ec, p2, nullptr, gen, tmp1, ctx);
        meter.measure([&]() {
            EC_POINT_add(ec, p3, p1, p2, ctx);
            return *((uint8_t *)p3);
        });
    };

    BENCHMARK_ADVANCED("secp256k1 scalar mul")
    (Catch::Benchmark::Chronometer meter) {
        BN_rand_range(tmp1, n);
        meter.measure([&]() {
            EC_POINT_mul(ec, p1, nullptr, gen, tmp1, ctx);
            return *((uint8_t *)p1);
        });
    };
}