#include "common.hpp"
#include "utils.hpp"

#include <gec/dlp.hpp>

#include "configured_catch.hpp"
#include "curve.hpp"

#include <iomanip>
#include <sstream>

using namespace gec;
using namespace dlp;

TEST_CASE("pollard_rho", "[dlp][pollard_rho]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;

    std::random_device rd;
    std::mt19937 rng(rd());

    C::Context<> ctx;

    C h;
    REQUIRE(C::on_curve(g, ctx));

    C::mul(h, S::mod(), g, ctx);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g, ctx);
    REQUIRE(C::on_curve(h, ctx));

    constexpr size_t l = 32;
    S al[l], bl[l];
    C pl[l];

    S c, d, mon_c, mon_d;

    pollard_rho(c, d, l, al, bl, pl, g, h, rng, ctx);
    S::to_montgomery(mon_c, c);
    S::to_montgomery(mon_d, d);
    S::inv(mon_d, ctx);
    S::mul(d, mon_c, mon_d);
    S::from_montgomery(c, d);

    C xg;
    C::mul(xg, c, g, ctx);
    CAPTURE(c, xg, h);
    REQUIRE(C::eq(xg, h));
}

TEST_CASE("pollard_rho bench", "[dlp][pollard_rho][bench]") {
    using S = Dlp3G1Scaler;

    std::random_device rd;
    std::mt19937 rng(rd());

    S x;
    S::sample_non_zero(x, rng);

    {
        using C = Dlp3CurveA;
        const C &g = Dlp3Gen1;

        C::Context<> ctx;

        C h;
        REQUIRE(C::on_curve(g, ctx));

        C::mul(h, S::mod(), g, ctx);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g, ctx);
        REQUIRE(C::on_curve(h, ctx));

        constexpr size_t l = 32;
        S al[l], bl[l];
        C pl[l];

        S c, d, mon_c, mon_d;

        BENCHMARK("pollard rho") {
            pollard_rho(c, d, l, al, bl, pl, g, h, rng, ctx);
            S::to_montgomery(mon_c, c);
            S::to_montgomery(mon_d, d);
            S::inv(mon_d, ctx);
            S::mul(d, mon_c, mon_d);
            S::from_montgomery(c, d);
            return c.array()[0];
        };
    }

#ifdef GEC_ENABLE_AVX2
    {
        using C = AVX2Dlp3CurveA;
        const C &g = reinterpret_cast<const C &>(Dlp3Gen1);

        C::Context<> ctx;

        C h;
        REQUIRE(C::on_curve(g, ctx));

        C::mul(h, S::mod(), g, ctx);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g, ctx);
        REQUIRE(C::on_curve(h, ctx));

        constexpr size_t l = 32;
        S al[l], bl[l];
        C pl[l];

        S c, d, mon_c, mon_d;

        BENCHMARK("avx2 pollard rho") {
            pollard_rho(c, d, l, al, bl, pl, g, h, rng, ctx);
            S::to_montgomery(mon_c, c);
            S::to_montgomery(mon_d, d);
            S::inv(mon_d, ctx);
            S::mul(d, mon_c, mon_d);
            S::from_montgomery(c, d);
            return c.array()[0];
        };
    }
#endif // GEC_ENABLE_AVX2
}

#ifdef GEC_ENABLE_PTHREAD

TEST_CASE("multithread_pollard_rho", "[dlp][pollard_rho][multithread]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;
    using F = C::Field;

    std::random_device rd;
    std::mt19937 rng(rd());

    C::Context<> ctx;

    C h;
    REQUIRE(C::on_curve(g, ctx));

    C::mul(h, S::mod(), g, ctx);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g, ctx);
    REQUIRE(C::on_curve(h, ctx));

    size_t l = 32;
    size_t worker_n = 8;
    S c, d, mon_c, mon_d;

    F mask(0x80000000, 0, 0, 0, 0, 0, 0, 0);

    multithread_pollard_rho(c, d, l, worker_n, mask, g, h);
    S::to_montgomery(mon_c, c);
    S::to_montgomery(mon_d, d);
    S::inv(mon_d, ctx);
    S::mul(d, mon_c, mon_d);
    S::from_montgomery(c, d);

    C xg;
    C::mul(xg, c, g, ctx);
    CAPTURE(c, xg, h);
    REQUIRE(C::eq(xg, h));
}

TEST_CASE("multithread_pollard_rho bench",
          "[dlp][pollard_rho][multithread][bench]") {
    using S = Dlp3G1Scaler;

    std::random_device rd;
    std::mt19937 rng(rd());
    S x;
    S::sample_non_zero(x, rng);

    {
        using C = AVX2Dlp3CurveA;
        const C &g = reinterpret_cast<const C &>(Dlp3Gen1);
        using F = C::Field;

        C::Context<> ctx;

        C h;
        REQUIRE(C::on_curve(g, ctx));

        C::mul(h, S::mod(), g, ctx);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g, ctx);
        REQUIRE(C::on_curve(h, ctx));

        size_t l = 32;
        size_t worker_n = 8;
        S c, d, mon_c, mon_d;

        std::basic_stringstream<char> bench_name;
        bench_name << "multithread pollard rho, mask 0x";

        for (int k = 0; k < 12; ++k) {
            F::LimbT m =
                F::LimbT(std::make_signed_t<F::LimbT>(0x80000000) >> k);

            bench_name.seekp(32);
            bench_name << std::hex << setw(8) << setfill('0') << m;

            BENCHMARK(bench_name.str()) {
                F mask(m, 0, 0, 0, 0, 0, 0, 0);
                multithread_pollard_rho(c, d, l, worker_n, mask, g, h);
                S::to_montgomery(mon_c, c);
                S::to_montgomery(mon_d, d);
                S::inv(mon_d, ctx);
                S::mul(d, mon_c, mon_d);
                S::from_montgomery(c, d);
                return c.array()[0];
            };
        }
    }

#ifdef GEC_ENABLE_AVX2
    {
        using C = AVX2Dlp3CurveA;
        const C &g = reinterpret_cast<const C &>(Dlp3Gen1);
        using F = C::Field;

        C::Context<> ctx;

        C h;
        REQUIRE(C::on_curve(g, ctx));

        C::mul(h, S::mod(), g, ctx);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g, ctx);
        REQUIRE(C::on_curve(h, ctx));

        size_t l = 32;
        size_t worker_n = 8;
        S c, d, mon_c, mon_d;

        std::basic_stringstream<char> bench_name;
        bench_name << "avx2 multithread pollard rho, mask 0x";

        for (int k = 0; k < 12; ++k) {
            F::LimbT m =
                F::LimbT(std::make_signed_t<F::LimbT>(0x80000000) >> k);

            bench_name.seekp(37);
            bench_name << std::hex << setw(8) << setfill('0') << m;

            BENCHMARK(bench_name.str()) {
                F mask(m, 0, 0, 0, 0, 0, 0, 0);
                multithread_pollard_rho(c, d, l, worker_n, mask, g, h);
                S::to_montgomery(mon_c, c);
                S::to_montgomery(mon_d, d);
                S::inv(mon_d, ctx);
                S::mul(d, mon_c, mon_d);
                S::from_montgomery(c, d);
                return c.array()[0];
            };
        }
    }
#endif // GEC_ENABLE_AVX2
}

#endif // GEC_ENABLE_PTHREAD
