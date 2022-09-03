#include <configured_catch.hpp>

#include <common.hpp>
#include <utils.hpp>

#include <gec/curve/secp256k1.hpp>

#include <random>

using namespace gec;
using namespace curve;
using namespace std;

TEST_CASE("lift_x", "[curve][jacobian]") {
    using C = secp256k1::Curve<>;
    using F = secp256k1::Field;
    using S = secp256k1::Scaler;
    using secp256k1::Gen;

    F x;
    C p1, p2;
    S s;

    std::random_device rd;
    auto data_seed = rd();
    auto seed = rd();
    CAPTURE(data_seed, seed);

    auto data_rng = make_gec_rng(std::mt19937(data_seed));
    std::uniform_int_distribution<uint8_t> gen;

    auto rng = make_gec_rng(std::mt19937(seed));

    for (int k = 0; k < 500; ++k) {
        S::sample(s, data_rng);
        C::mul(p1, s, Gen);
        C::to_affine(p1);
        CAPTURE(p1);
        REQUIRE(C::lift_x(p2, p1.x(), p1.y().array()[0] & 0x1, rng));
        CAPTURE(p2);
        REQUIRE(p1.x() == p2.x());
        REQUIRE(p1.y() == p2.y());
    }

    for (int k = 0; k < 500; ++k) {
        F::sample(x, data_rng);
        bool bit = gen(data_rng.get_rng()) & 0x1;
        CAPTURE(x, bit);

        C::lift_x_with_inc(p2, x, bit, rng);

        REQUIRE(C::on_curve(p2));
        REQUIRE((p2.y().array()[0] & 0x1) == bit);
    }
}