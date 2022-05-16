#include "common.hpp"
#include "utils.hpp"

#include <gec/utils/misc.hpp>

#include <random>

#include "configured_catch.hpp"

using namespace gec;
using namespace utils;

TEST_CASE("trailing_zeros", "[utils][misc]") {
    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_int_distribution<> gen;

    for (int k = 0; k < 32; ++k) {
        unsigned int x = 1 << k;
        REQUIRE(k == trailing_zeros(x));
    }

    for (int k = 0; k < 1000; ++k) {
        int x = gen(rng);
        auto n = trailing_zeros(x);
        CAPTURE(x, n);
        REQUIRE((x & -x) == (1 << n));

        unsigned int ux = x;
        auto un = trailing_zeros(ux);
        CAPTURE(ux, un);
        REQUIRE((x & -x) == (1 << un));
    }
}