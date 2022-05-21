#include "common.hpp"

#include <gec/bigint.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

class alignas(32) AddGroup
    : public ArrayBE<LIMB_T, LN_256>,
      public VtCompare<AddGroup, LIMB_T, LN_256>,
      public BitOps<AddGroup, LIMB_T, LN_256>,
      public ModAddSub<AddGroup, LIMB_T, LN_256, MOD_256>,
      public ArrayOstream<AddGroup, LIMB_T, LN_256>,
      public ModRandom<AddGroup, LIMB_T, LN_256, MOD_256>,
      public ArrayPrint<AddGroup, LIMB_T, LN_256> {
  public:
    using ArrayBE::ArrayBE;
};

class Field : public Array<LIMB_T, LN_256>,
              public Constants<Field, LIMB_T, LN_256>,
              public Montgomery<Field, LIMB_T, LN_256, MOD_256, MOD_P_256,
                                RR_256, OneR_256> {
  public:
    using Array::Array;
};

class alignas(32) AVX2Field
    : public Array<LIMB_T, LN_256>,
      public VtCompare<AVX2Field, LIMB_T, LN_256>,
      public Constants<AVX2Field, LIMB_T, LN_256>,
      public AVX2Montgomery<AVX2Field, LIMB_T, LN_256, MOD_256, MOD_P_256,
                            RR_256, OneR_256> {
  public:
    using Array::Array;
};

TEST_CASE("avx2 montgomery", "[ring][field]") {
    using gec::utils::CmpEnum;
    using gec::utils::VtSeqCmp;

    std::random_device rd;
    std::mt19937 rng(rd());

    AddGroup x_arr(0x1f82f372u, 0x62639538u, 0xca640ff9u, 0xed12396au,
                   0x9c4d50dau, 0xff21e339u, 0xfbfa64d8u, 0x75b40000u);
    AddGroup y_arr(0xed469d79u, 0xaba8d6fau, 0x6724432cu, 0x7221f040u,
                   0x6416351du, 0x923ec2cau, 0x72bc1127u, 0xf1e018aau);
    AddGroup mon_x_arr, mon_y_arr, mon_xy_arr, xy_arr;
    AddGroup one_arr(1);

    for (int k = 0; k < 10000; ++k) {
        AddGroup::sample(x_arr, rng);
        AddGroup::sample(y_arr, rng);
        CAPTURE(AddGroup::mod(), x_arr, y_arr);

        {
            using F = Field;
            const auto &x = reinterpret_cast<F &>(x_arr);
            const auto &y = reinterpret_cast<F &>(y_arr);
            auto &mon_x = reinterpret_cast<F &>(mon_x_arr);
            auto &mon_y = reinterpret_cast<F &>(mon_y_arr);
            auto &mon_xy = reinterpret_cast<F &>(mon_xy_arr);
            auto &xy = reinterpret_cast<F &>(xy_arr);

            F::to_montgomery(mon_x, x);
            F::to_montgomery(mon_y, y);
            F::mul(mon_xy, mon_x, mon_y);
            F::from_montgomery(xy, mon_xy);
        }
        CAPTURE(mon_x_arr, mon_y_arr);
        CAPTURE(mon_xy_arr, xy_arr);

        {
            using F = AVX2Field;
            const auto &x = reinterpret_cast<F &>(x_arr);
            const auto &y = reinterpret_cast<F &>(y_arr);
            const auto &expected_mon_x = reinterpret_cast<F &>(mon_x_arr);
            const auto &expected_mon_y = reinterpret_cast<F &>(mon_y_arr);
            const auto &expected_mon_xy = reinterpret_cast<F &>(mon_xy_arr);
            const auto &expected_xy = reinterpret_cast<F &>(xy_arr);

            F mon_x, mon_y, mon_xy, xy;
            F::to_montgomery(mon_x, x);
            CAPTURE(mon_x);
            REQUIRE(expected_mon_x == mon_x);
            F::to_montgomery(mon_y, y);
            CAPTURE(mon_y);
            REQUIRE(expected_mon_y == mon_y);
            F::mul(mon_xy, mon_x, mon_y);
            CAPTURE(mon_xy);
            REQUIRE(expected_mon_xy == mon_xy);
            F::from_montgomery(xy, mon_xy);
            CAPTURE(xy);
            REQUIRE(expected_xy == xy);
        }
    }
}

TEST_CASE("256 montgomery bench", "[ring][field][bench]") {
    std::random_device rd;
    std::mt19937 rng(rd());

    AddGroup one_arr(1);
    AddGroup x_arr, y_arr;
    AddGroup mon_x_arr, mon_y_arr;

    AddGroup::sample(x_arr, rng);
    AddGroup::sample(y_arr, rng);

    {
        using F = Field;
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &y = reinterpret_cast<const F &>(y_arr);
        auto &mon_x = reinterpret_cast<F &>(mon_x_arr);
        auto &mon_y = reinterpret_cast<F &>(mon_y_arr);

        F::to_montgomery(mon_x, x);
        F::to_montgomery(mon_y, y);
    }

    {
        using F = Field;
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &mon_x = reinterpret_cast<const F &>(mon_x_arr);
        const auto &mon_y = reinterpret_cast<const F &>(mon_y_arr);

        BENCHMARK("into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
            return res;
        };

        BENCHMARK("montgomery mul") {
            F mon_xy;
            F::mul(mon_xy, mon_x, mon_y);
            return mon_xy;
        };
    }

    {
        using F = AVX2Field;
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &mon_x = reinterpret_cast<const F &>(mon_x_arr);
        const auto &mon_y = reinterpret_cast<const F &>(mon_y_arr);

        BENCHMARK("avx2 into montgomery form") {
            F res;
            F::to_montgomery(res, x);
            return res;
        };

        BENCHMARK("avx2 from montgomery form") {
            F res;
            F::from_montgomery(res, mon_x);
            return res;
        };

        BENCHMARK("avx2 montgomery mul") {
            F mon_xy;
            F::mul(mon_xy, mon_x, mon_y);
            return mon_xy;
        };
    }
}
