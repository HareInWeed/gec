#pragma once
#ifndef GEC_BIGINT_MIXIN_RANDOM_HPP
#define GEC_BIGINT_MIXIN_RANDOM_HPP

#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/sequence.hpp>

#include <random>

namespace gec {

namespace bigint {

/** @brief mixin that enables exponentiation
 */
template <class Core, typename LIMB_T, size_t LIMB_N, const LIMB_T *MOD>
class ModRandom : protected CRTP<Core, ModRandom<Core, LIMB_T, LIMB_N, MOD>> {
    friend CRTP<Core, ModRandom<Core, LIMB_T, LIMB_N, MOD>>;

  public:
    template <typename Rng>
    __host__ __device__ GEC_INLINE static void sample(Core &GEC_RSTRCT a,
                                                      Rng &rng) {
        sample_exclusive_raw(a, MOD, rng);
    }
    template <typename Rng>
    __host__ __device__ GEC_INLINE static void
    sample(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT upper, Rng &rng) {
        sample_exclusive_raw(a, upper.array(), rng);
    }
    template <typename Rng, typename Ctx>
    __host__ __device__ GEC_INLINE static void
    sample(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT lower,
           const Core &GEC_RSTRCT upper, Rng &rng, Ctx &ctx) {
        auto &ctx_view = ctx.template view_as<Core>();
        auto &span = ctx_view.template get<0>();
        Core::sub(span, upper, lower);
        do {
            sample_inclusive(a, span, rng);
        } while (utils::VtSeqAll<LIMB_N, LIMB_T, utils::ops::Eq<LIMB_T>>::call(
            a.array(), span.array()));
        Core::add(a, lower);
    }

    template <typename Rng>
    __host__ __device__ GEC_INLINE static void
    sample_inclusive(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT upper,
                     Rng &rng) {
        sample_inclusive_raw(a, upper.array(), rng);
    }
    template <typename Rng, typename Ctx>
    __host__ __device__ GEC_INLINE static void
    sample_inclusive(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT lower,
                     const Core &GEC_RSTRCT upper, Rng &rng, Ctx &ctx) {
        auto &ctx_view = ctx.template view_as<Core>();
        auto &span = ctx_view.template get<0>();
        Core::sub(span, upper, lower);
        sample_inclusive(a, span, rng);
        Core::add(a, lower);
    }

    template <typename Rng>
    __host__ __device__ static void
    sample_inclusive_raw(Core &GEC_RSTRCT a, const LIMB_T *GEC_RSTRCT bound,
                         Rng &rng) {
        bool is_gt;
        bool is_eq;
        do {
            is_eq = true;
            is_gt = false;
            bool leading_zero = true;
            int k = LIMB_N;

            do {
                --k;

                LIMB_T mask = ~LIMB_T(0);
                if (leading_zero) {
                    mask = utils::significant_mask(bound[k]);
                    leading_zero = mask == 0;
                }

                std::uniform_int_distribution<LIMB_T> gen(0, mask);
                a.array()[k] = gen(rng);
                if (is_eq) {
                    if (a.array()[k] > bound[k]) {
                        is_gt = true;
                        break;
                    } else if (a.array()[k] < bound[k]) {
                        is_eq = false;
                    }
                }
            } while (k != 0);
        } while (is_gt);
    }

    template <typename Rng>
    __host__ __device__ static void
    sample_exclusive_raw(Core &GEC_RSTRCT a, const LIMB_T *GEC_RSTRCT bound,
                         Rng &rng) {
        bool is_ge;
        do {
            is_ge = true;
            bool leading_zero = true;
            int k = LIMB_N;

            do {
                --k;

                LIMB_T mask = ~LIMB_T(0);
                if (leading_zero) {
                    mask = utils::significant_mask(bound[k]);
                    leading_zero = mask == 0;
                }

                std::uniform_int_distribution<LIMB_T> gen(0, mask);
                a.array()[k] = gen(rng);
                if (is_ge && a.array()[k] > bound[k]) {
                    is_ge = true;
                    break;
                }
                is_ge = is_ge && (a.array()[k] == bound[k]);
            } while (k != 0);
        } while (is_ge);
    }

    template <typename Rng>
    __host__ __device__ static void sample_non_zero(Core &GEC_RSTRCT a,
                                                    Rng &rng) {
        do {
            sample(a, rng);
        } while (a.is_zero());
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_RANDOM_HPP
