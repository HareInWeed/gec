#include <configured_catch.hpp>

#include <gec/curve/secp256k1.hpp>

TEST_CASE("gec_cpu", "[gec][bench]") {
    using namespace gec;
    using S = curve::secp256k1::Scalar;
    using C = curve::secp256k1::Curve<curve::ProjectiveCurve>;
    using curve::secp256k1::Gen;

    std::random_device rd;
    auto seed = rd();
    auto rng = make_gec_rng(std::mt19937(seed));

    const C G{Gen.x(), Gen.y(), Gen.z()};

    S s1, s2;
    C p1, p2, p3;

    BENCHMARK_ADVANCED("secp256k1 point add")
    (Catch::Benchmark::Chronometer meter) {
        S::sample(s1, rng);
        C::mul(p1, s1, G);
        S::sample(s2, rng);
        C::mul(p2, s2, G);
        meter.measure([&]() {
            C::add(p3, p1, p2);
            return &p3;
        });
    };

    BENCHMARK_ADVANCED("secp256k1 scalar mul")
    (Catch::Benchmark::Chronometer meter) {
        S::sample(s1, rng);
        meter.measure([&]() {
            C::mul(p1, s1, G);
            return &p1;
        });
    };
}