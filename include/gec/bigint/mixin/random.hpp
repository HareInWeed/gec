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
    template <typename RngEngine>
    __host__ __device__ static void sample(Core &GEC_RSTRCT a, RngEngine &rng) {
        bool is_ge;
        do {
            is_ge = true;
            bool leading_zero = true;
            int k = LIMB_N;

            do {
                --k;

                LIMB_T mask = ~LIMB_T(0);
                if (leading_zero) {
                    mask = utils::significant_mask(MOD[k]);
                    leading_zero = mask == 0;
                }

                std::uniform_int_distribution<LIMB_T> gen(0, mask);
                a.array()[k] = gen(rng);
                if (is_ge && a.array()[k] > MOD[k]) {
                    is_ge = true;
                    break;
                }
                is_ge = is_ge && (a.array()[k] == MOD[k]);
            } while (k != 0);
        } while (is_ge);
    }

    template <typename RngEngine>
    __host__ __device__ static void sample_non_zero(Core &GEC_RSTRCT a,
                                                    RngEngine &rng) {
        do {
            sample(a, rng);
        } while (a.is_zero());
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_RANDOM_HPP
