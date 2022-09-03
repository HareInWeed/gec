#include <common.hpp>
#include <utils.hpp>

// #define GEC_DEBUG
#include <gec/dlp.hpp>

#include <configured_catch.hpp>

#ifdef GEC_NVCC
GEC_NV_DIAGNOSTIC_PUSH
// disable NULL reference is not allowed warning, this occurs because
// all device pointer in template parameters of AVX2Field are set to `nullptr`
GEC_NV_DIAG_SUPPRESS(284)
GEC_CALL_H_FROM_H_D
#endif // GEC_NVCC
#include "curve.hpp"
#ifdef GEC_NVCC
GEC_NV_DIAGNOSTIC_POP
#endif // GEC_NVCC

#include <iomanip>
#include <sstream>

using namespace gec;
using namespace dlp;

TEST_CASE("pollard_rho", "[dlp][pollard_rho]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C h;
    REQUIRE(C::on_curve(g));

    C::mul(h, S::mod(), g);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g);
    REQUIRE(C::on_curve(h));

    constexpr size_t l = 16;
    S al[l], bl[l];
    C pl[l];

    S c, d, mon_c, mon_d;

    pollard_rho(c, d, l, al, bl, pl, g, h, rng);
    S::to_montgomery(mon_c, c);
    S::to_montgomery(mon_d, d);
    S::inv(mon_d);
    S::mul(d, mon_c, mon_d);
    S::from_montgomery(c, d);

    C xg;
    C::mul(xg, c, g);
    CAPTURE(c, xg, h);
    REQUIRE(C::eq(xg, h));
}

TEST_CASE("pollard_rho bench", "[dlp][pollard_rho][bench]") {
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    S x;
    S::sample_non_zero(x, rng);

    {
        using C = Dlp3CurveA;
        const C &g = Dlp3Gen1;

        C h;
        REQUIRE(C::on_curve(g));

        C::mul(h, S::mod(), g);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g);
        REQUIRE(C::on_curve(h));

        constexpr size_t l = 16;
        S al[l], bl[l];
        C pl[l];

        S c, d, mon_c, mon_d;

        BENCHMARK("pollard rho") {
            pollard_rho(c, d, l, al, bl, pl, g, h, rng);
            S::to_montgomery(mon_c, c);
            S::to_montgomery(mon_d, d);
            S::inv(mon_d);
            S::mul(d, mon_c, mon_d);
            S::from_montgomery(c, d);
            return c.array()[0];
        };
    }

#ifdef GEC_ENABLE_AVX2
    {
        using C = AVX2Dlp3CurveA;
        const C &g = reinterpret_cast<const C &>(Dlp3Gen1);

        C h;
        REQUIRE(C::on_curve(g));

        C::mul(h, S::mod(), g);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g);
        REQUIRE(C::on_curve(h));

        constexpr size_t l = 16;
        S al[l], bl[l];
        C pl[l];

        S c, d, mon_c, mon_d;

        BENCHMARK("avx2 pollard rho") {
            pollard_rho(c, d, l, al, bl, pl, g, h, rng);
            S::to_montgomery(mon_c, c);
            S::to_montgomery(mon_d, d);
            S::inv(mon_d);
            S::mul(d, mon_c, mon_d);
            S::from_montgomery(c, d);
            return c.array()[0];
        };
    }
#endif // GEC_ENABLE_AVX2
}

TEST_CASE("pollard_lambda", "[dlp][pollard_lambda]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C h;
    REQUIRE(C::on_curve(g));

    C::mul(h, S::mod(), g);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x0, lower(1 << 3), upper((1 << 3) + (1 << 15)), bound(1 << 8);
    S::sample_inclusive(x0, lower, upper, rng);

    C::mul(h, x0, g);
    REQUIRE(C::on_curve(h));

    size_t l = 15;
    std::vector<S> sl(l);
    std::vector<C> pl(l);

    S x;

    pollard_lambda(x, sl.data(), pl.data(), bound, lower, upper, g, h, rng);

    C xg;
    C::mul(xg, x, g);
    CAPTURE(x, xg, h);
    REQUIRE(C::eq(xg, h));
}

#ifdef GEC_ENABLE_PTHREADS

TEST_CASE("multithread_pollard_rho", "[dlp][pollard_rho][multithread]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;
    using F = C::Field;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C h;
    REQUIRE(C::on_curve(g));

    C::mul(h, S::mod(), g);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g);
    REQUIRE(C::on_curve(h));

    size_t l = 16;
    size_t worker_n = 8;
    S c, d, mon_c, mon_d;

    F mask(0x80000000, 0, 0, 0, 0, 0, 0, 0);

    multithread_pollard_rho(c, d, l, worker_n, mask, g, h, rng);
    S::to_montgomery(mon_c, c);
    S::to_montgomery(mon_d, d);
    S::inv(mon_d);
    S::mul(d, mon_c, mon_d);
    S::from_montgomery(c, d);

    C xg;
    C::mul(xg, c, g);
    CAPTURE(c, xg, h);
    REQUIRE(C::eq(xg, h));
}

TEST_CASE("multithread_pollard_rho bench",
          "[dlp][pollard_rho][multithread][bench]") {
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));
    S x;
    S::sample_non_zero(x, rng);

    {
        using C = Dlp3CurveA;
        const C &g = reinterpret_cast<const C &>(Dlp3Gen1);
        using F = C::Field;

        C h;
        REQUIRE(C::on_curve(g));

        C::mul(h, S::mod(), g);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g);
        REQUIRE(C::on_curve(h));

        size_t l = 16;
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
                multithread_pollard_rho(c, d, l, worker_n, mask, g, h, rng);
                S::to_montgomery(mon_c, c);
                S::to_montgomery(mon_d, d);
                S::inv(mon_d);
                S::mul(d, mon_c, mon_d);
                S::from_montgomery(c, d);
                return c.array()[0];
            };
        }
    }

#ifdef GEC_ENABLE_AVX2

    {
        using C = AVX2Dlp3CurveA;
        const C &g = *reinterpret_cast<const C *>(Dlp3Gen1.array());
        using F = C::Field;

        C h;
        REQUIRE(C::on_curve(g));

        C::mul(h, S::mod(), g);
        CAPTURE(h);
        REQUIRE(h.is_inf());

        C::mul(h, x, g);
        REQUIRE(C::on_curve(h));

        size_t l = 16;
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
                multithread_pollard_rho(c, d, l, worker_n, mask, g, h, rng);
                S::to_montgomery(mon_c, c);
                S::to_montgomery(mon_d, d);
                S::inv(mon_d);
                S::mul(d, mon_c, mon_d);
                S::from_montgomery(c, d);
                return c.array()[0];
            };
        }
    }
#endif // GEC_ENABLE_AVX2
}

TEST_CASE("multithread_pollard_lambda", "[dlp][pollard_lambda][multithread]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    CAPTURE(seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C h;
    REQUIRE(C::on_curve(g));

    C::mul(h, S::mod(), g);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x0, lower(1 << 3), upper((1 << 3) + (1 << 15)), bound(1 << 5);
    S::sample_inclusive(x0, lower, upper, rng);

    C::mul(h, x0, g);
    REQUIRE(C::on_curve(h));

    S x;
    multithread_pollard_lambda(x, bound, 8, lower, upper, g, h, rng);

    C xg;
    C::mul(xg, x, g);
    CAPTURE(x, xg, h);
    REQUIRE(C::eq(xg, h));
}

TEST_CASE("multithread_pollard_lambda bench",
          "[dlp][pollard_lambda][multithread][bench]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C h;
    REQUIRE(C::on_curve(g));

    C::mul(h, S::mod(), g);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x0, lower(1 << 3), upper((1 << 3) + (1 << 15)), bound(1 << 5);
    S::sample_inclusive(x0, lower, upper, rng);

    C::mul(h, x0, g);
    REQUIRE(C::on_curve(h));

    size_t l = 15;
    std::vector<S> sl(l);
    std::vector<C> pl(l);

    S x;

    BENCHMARK("multithread_pollard_lambda") {
        multithread_pollard_lambda(x, bound, 8, lower, upper, g, h, rng);
        return x;
    };
}

#endif // GEC_ENABLE_PTHREADS
