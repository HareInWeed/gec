#include "common.hpp"

#include <gec/bigint.hpp>
#include <gec/bigint/mixin/ostream.hpp>
#include <gec/bigint/mixin/print.hpp>

#include "configured_catch.hpp"

using namespace gec;
using namespace bigint;

class alignas(32) AddGroup
    : public Array<LIMB_T, LN_256>,
      public VtCompareMixin<AddGroup, LIMB_T, LN_256>,
      public BitOpsMixin<AddGroup, LIMB_T, LN_256>,
      public ModAddSubMixin<AddGroup, LIMB_T, LN_256, MOD_256>,
      public ArrayOstreamMixin<AddGroup, LIMB_T, LN_256>,
      public ArrayPrintMixin<AddGroup, LIMB_T, LN_256> {
  public:
    using Array::Array;
};

class Field
    : public Array<uint64_t, 4>,
      public Montgomery<Field, uint64_t, 4,
                        reinterpret_cast<const uint64_t (&)[4]>(MOD_256),
                        0xd838091dd2253531u> {
  public:
    using Array::Array;
};

class AVX2Field
    : public Array<LIMB_T, LN_256>,
      public VtCompareMixin<AVX2Field, LIMB_T, LN_256>,
      public AVX2Montgomery<AVX2Field, LIMB_T, LN_256, MOD_256, MOD_P_256> {
  public:
    using Array::Array;
};

TEST_CASE("avx montgomery", "[ring][field]") {
    using gec::utils::CmpEnum;
    using gec::utils::VtSeqCmp;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    const auto &MOD = reinterpret_cast<const AddGroup &>(MOD_256);
    AddGroup x_arr(0x1f82f372u, 0x62639538u, 0xca640ff9u, 0xed12396au,
                   0x9c4d50dau, 0xff21e339u, 0xfbfa64d8u, 0x75b40000u);
    AddGroup y_arr(0xed469d79u, 0xaba8d6fau, 0x6724432cu, 0x7221f040u,
                   0x6416351du, 0x923ec2cau, 0x72bc1127u, 0xf1e018aau);
    AddGroup xy_arr;
    AddGroup one_arr(1);

    for (int k = 0; k < 1000000; ++k) {
        do {
            for (int i = 0; i < LN_256; ++i) {
                x_arr.get_arr()[i] = dis_u32(gen);
            }
        } while (x_arr > MOD);
        do {
            for (int i = 0; i < LN_256; ++i) {
                y_arr.get_arr()[i] = dis_u32(gen);
            }
        } while (y_arr > MOD);

        // x_arr.println();
        // y_arr.println();
        // printf("\n");

        {
            using F = Field;
            const auto &RR = reinterpret_cast<const F &>(RR_256);
            const auto &One = reinterpret_cast<const F &>(one_arr);
            const auto &x = reinterpret_cast<const F &>(x_arr);
            const auto &y = reinterpret_cast<const F &>(y_arr);
            auto &xy = reinterpret_cast<F &>(xy_arr);

            F mon_x, mon_y, mon_xy;
            F::mul(mon_x, x, RR);
            // mon_x.println();
            F::mul(mon_y, y, RR);
            // mon_y.println();
            F::mul(mon_xy, mon_x, mon_y);
            // mon_xy.println();
            F::mul(xy, mon_xy, One);
            // xy.println();
            // printf("\n");
        }

        {
            using F = AVX2Field;
            const auto &RR = reinterpret_cast<const F &>(RR_256);
            const auto &One = reinterpret_cast<const F &>(one_arr);
            const auto &x = reinterpret_cast<const F &>(x_arr);
            const auto &y = reinterpret_cast<const F &>(y_arr);
            const auto &expected_xy = reinterpret_cast<F &>(xy_arr);

            F mon_x, mon_y, mon_xy, xy;
            F::mul(mon_x, x, RR);
            // mon_x.println();
            F::mul(mon_y, y, RR);
            // mon_y.println();
            F::mul(mon_xy, mon_x, mon_y);
            // mon_xy.println();
            F::mul(xy, mon_xy, One);
            // xy.println();
            // printf("\n");
            REQUIRE(expected_xy == xy);
        }
    }
}

TEST_CASE("256 montgomery bench", "[ring][field][bench]") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<LIMB_T> dis_u32(
        std::numeric_limits<LIMB_T>::min(), std::numeric_limits<LIMB_T>::max());

    const auto &MOD = reinterpret_cast<const AddGroup &>(MOD_256);
    AddGroup one_arr(1);
    AddGroup x_arr;
    AddGroup y_arr;
    AddGroup mon_x_arr;
    AddGroup mon_y_arr;
    do {
        for (int i = 0; i < LN_256; ++i) {
            x_arr.get_arr()[i] = dis_u32(gen);
        }
    } while (x_arr > MOD);
    do {
        for (int i = 0; i < LN_256; ++i) {
            y_arr.get_arr()[i] = dis_u32(gen);
        }
    } while (y_arr > MOD);

    {
        using F = Field;
        const auto &RR = reinterpret_cast<const F &>(RR_256);
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &y = reinterpret_cast<const F &>(y_arr);
        auto &mon_x = reinterpret_cast<F &>(mon_x_arr);
        auto &mon_y = reinterpret_cast<F &>(mon_y_arr);

        F::mul(mon_x, x, RR);
        F::mul(mon_y, y, RR);
    }

    {
        using F = Field;
        const auto &RR = reinterpret_cast<const F &>(RR_256);
        const auto &One = reinterpret_cast<const F &>(one_arr);
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &y = reinterpret_cast<const F &>(y_arr);
        const auto &mon_x = reinterpret_cast<const F &>(mon_x_arr);
        const auto &mon_y = reinterpret_cast<const F &>(mon_y_arr);

        BENCHMARK("into montgomery form") {
            F res;
            F::mul(res, x, RR);
            return res;
        };

        BENCHMARK("from montgomery form") {
            F res;
            F::mul(res, mon_x, One);
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
        const auto &RR = reinterpret_cast<const F &>(RR_256);
        const auto &One = reinterpret_cast<const F &>(one_arr);
        const auto &x = reinterpret_cast<const F &>(x_arr);
        const auto &y = reinterpret_cast<const F &>(y_arr);
        const auto &mon_x = reinterpret_cast<const F &>(mon_x_arr);
        const auto &mon_y = reinterpret_cast<const F &>(mon_y_arr);

        BENCHMARK("avx into montgomery form") {
            F res;
            F::mul(res, x, RR);
            return res;
        };

        BENCHMARK("avx from montgomery form") {
            F res;
            F::mul(res, mon_x, One);
            return res;
        };

        BENCHMARK("avx montgomery mul") {
            F mon_xy;
            F::mul(mon_xy, mon_x, mon_y);
            return mon_xy;
        };
    }
}
