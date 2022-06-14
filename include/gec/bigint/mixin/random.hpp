#pragma once
#ifndef GEC_BIGINT_MIXIN_RANDOM_HPP
#define GEC_BIGINT_MIXIN_RANDOM_HPP

#include <gec/utils/crtp.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/sequence.hpp>

#include <random>

#ifdef __CUDACC__
#include <cassert>
#include <curand_kernel.h>
#include <thrust/random.h>
#endif // __CUDACC__

namespace gec {

namespace _gec_rng_ {
struct rng_enable {
    static const bool value = true;
};

template <typename Rng>
struct is_std_rng {
    static const bool value = false;
};

template <class UIntType, UIntType a, UIntType c, UIntType m>
struct is_std_rng<std::linear_congruential_engine<UIntType, a, c, m>>
    : public rng_enable {};
template <class UIntType, std::size_t w, std::size_t n, std::size_t m,
          std::size_t r, UIntType a, std::size_t u, UIntType d, std::size_t s,
          UIntType b, std::size_t t, UIntType c, std::size_t l, UIntType f>
struct is_std_rng<std::mersenne_twister_engine<UIntType, w, n, m, r, a, u, d, s,
                                               b, t, c, l, f>>
    : public rng_enable {};
template <class UIntType, std::size_t w, std::size_t s, std::size_t r>
struct is_std_rng<std::subtract_with_carry_engine<UIntType, w, s, r>>
    : public rng_enable {};
template <class Engine, std::size_t P, std::size_t R>
struct is_std_rng<std::discard_block_engine<Engine, P, R>> : public rng_enable {
};
template <class Engine, std::size_t W, class UIntType>
struct is_std_rng<std::independent_bits_engine<Engine, W, UIntType>>
    : public rng_enable {};
template <class Engine, std::size_t K>
struct is_std_rng<std::shuffle_order_engine<Engine, K>> : public rng_enable {};
template <>
struct is_std_rng<std::random_device> : public rng_enable {};

#ifdef __CUDACC__

template <typename Rng>
struct is_cu_rand_rng {
    static const bool value = false;
};

template <>
struct is_cu_rand_rng<curandStateMtgp32_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateSobol64_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateScrambledSobol32_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateScrambledSobol64_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateSobol32_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateMRG32k3a_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStatePhilox4_32_10_t> : public rng_enable {};
template <>
struct is_cu_rand_rng<curandStateXORWOW_t> : public rng_enable {};

template <typename T>
struct MaxHelper {
    static const T value = std::numeric_limits<T>::max();
};

template <typename Rng>
struct is_thrust_rng {
    static const bool value = false;
};

template <typename UIntType, UIntType a, UIntType c, UIntType m>
struct is_thrust_rng<
    thrust::random::linear_congruential_engine<UIntType, a, c, m>>
    : public rng_enable {};
template <typename UIntType, size_t w, size_t k, size_t q, size_t s>
struct is_thrust_rng<
    thrust::random::linear_feedback_shift_engine<UIntType, w, k, q, s>>
    : public rng_enable {};
template <typename UIntType, size_t w, size_t s, size_t r>
struct is_thrust_rng<
    thrust::random::subtract_with_carry_engine<UIntType, w, s, r>>
    : public rng_enable {};
template <typename Engine, size_t p, size_t r>
struct is_thrust_rng<thrust::random::discard_block_engine<Engine, p, r>>
    : public rng_enable {};
template <typename Engine1, size_t s1, typename Engine2, size_t s2>
struct is_thrust_rng<
    thrust::random::xor_combine_engine<Engine1, s1, Engine2, s2>>
    : public rng_enable {};

#endif // __CUDACC__

} // namespace _gec_rng_

template <typename Rng, typename Enable = void>
struct GecRng {
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &lower, const T &higher);
    template <typename T>
    __host__ __device__ GEC_INLINE T sample();
    __host__ __device__ GEC_INLINE Rng &get_rng();
};

template <typename Rng>
class GecRng<Rng, std::enable_if_t<_gec_rng_::is_std_rng<Rng>::value>> {
    Rng rng;

    template <typename T>
    __device__ GEC_INLINE T reject() {
        int std_rng_is_not_available_in_device_code = 0;
        assert(std_rng_is_not_available_in_device_code);
        return T(std_rng_is_not_available_in_device_code);
    }

  public:
    __host__ __device__ GEC_INLINE GecRng(Rng &&rng) : rng(std::move(rng)){};
    template <typename T>
    __host__ __device__ GEC_INLINE T sample() {
#ifdef __CUDA_ARCH__
        return reject<T>();
#else
        std::uniform_int_distribution<T> gen;
        return gen(rng);
#endif // __CUDA_ARCH__
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &higher) {
#ifdef __CUDA_ARCH__
        return reject<T>();
#else
        std::uniform_int_distribution<T> gen(T(0), higher);
        return gen(rng);
#endif // __CUDA_ARCH__
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &lower, const T &higher) {
#ifdef __CUDA_ARCH__
        return reject<T>();
#else
        std::uniform_int_distribution<T> gen(lower, higher);
        return gen(rng);
#endif // __CUDA_ARCH__
    }
    __host__ __device__ GEC_INLINE Rng &get_rng() { return rng; }
};

#ifdef __CUDACC__

template <typename Rng>
class GecRng<Rng, std::enable_if_t<_gec_rng_::is_cu_rand_rng<Rng>::value>> {
    Rng rng;

    template <typename T>
    __host__ GEC_INLINE T reject() {
        int cu_rand_rng_is_not_available_in_host_code = 0;
        assert(cu_rand_rng_is_not_available_in_host_code);
        return T(cu_rand_rng_is_not_available_in_host_code);
    }

  public:
    __host__ __device__ GEC_INLINE GecRng(Rng &&rng) : rng(std::move(rng)){};
    template <typename T>
    __host__ __device__ GEC_INLINE T sample() {
#ifdef __CUDA_ARCH__
        // FIXME: handle cases where the type of rng generation is not T
        return curand(&get_rng());
#else
        return reject<T>();
#endif // __CUDA_ARCH__
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &higher) {
#ifdef __CUDA_ARCH__
        // FIXME: handle cases where max of rng generated number is less than
        // higher
        constexpr T t_max = _gec_rng_::MaxHelper<T>::value;
        if (higher == t_max) {
            return this->template sample<T>();
        } else {
            T m = higher + 1;
            T bound = (t_max / m) * m;
            while (true) {
                T x = this->template sample<T>();
                if (x <= bound) {
                    return x % m;
                }
            }
        }
#else
        return reject<T>();
#endif // __CUDA_ARCH__
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &lower, const T &higher) {
#ifdef __CUDA_ARCH__
        return lower + sample(higher - lower);
#else
        return reject<T>();
#endif // __CUDA_ARCH__
    }
    __host__ __device__ GEC_INLINE Rng &get_rng() { return rng; }
};

template <typename Rng>
class GecRng<Rng, std::enable_if_t<_gec_rng_::is_thrust_rng<Rng>::value>> {
    Rng rng;

  public:
    __host__ __device__ GEC_INLINE GecRng(Rng &&rng) : rng(std::move(rng)){};
    template <typename T>
    __host__ __device__ GEC_INLINE T sample() {
        thrust::uniform_int_distribution<T> gen;
        return gen(rng);
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &higher) {
        thrust::uniform_int_distribution<T> gen(T(0), higher);
        return gen(rng);
    }
    template <typename T>
    __host__ __device__ GEC_INLINE T sample(const T &lower, const T &higher) {
        thrust::uniform_int_distribution<T> gen(lower, higher);
        return gen(rng);
    }
    __host__ __device__ GEC_INLINE Rng &get_rng() { return rng; }
};

#endif // __CUDACC__

template <typename Rng, typename... Args>
__host__ __device__ GEC_INLINE GecRng<Rng> make_gec_rng(Rng &&rng,
                                                        Args... args) {
    return GecRng<Rng>(std::forward<Rng>(rng), args...);
}

namespace bigint {

/** @brief mixin that enables exponentiation
 */
template <class Core, typename LIMB_T, size_t LIMB_N>
class ModRandom : protected CRTP<Core, ModRandom<Core, LIMB_T, LIMB_N>> {
    friend CRTP<Core, ModRandom<Core, LIMB_T, LIMB_N>>;

  public:
    template <typename Rng>
    __host__ __device__ GEC_INLINE static void sample(Core &GEC_RSTRCT a,
                                                      GecRng<Rng> &rng) {
        sample_exclusive_raw(a, a.mod().array(), rng);
    }
    template <typename Rng>
    __host__ __device__ GEC_INLINE static void
    sample(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT upper, GecRng<Rng> &rng) {
        sample_exclusive_raw(a, upper.array(), rng);
    }
    template <typename Rng, typename Ctx>
    __host__ __device__ GEC_INLINE static void
    sample(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT lower,
           const Core &GEC_RSTRCT upper, GecRng<Rng> &rng, Ctx &ctx) {
        auto &ctx_view = ctx.template view_as<Core>();
        auto &span = ctx_view.template get<0>();
        Core::sub(span, upper, lower);
        sample_exclusive_raw(a, span.array(), rng);
        Core::add(a, lower);
    }

    template <typename Rng>
    __host__ __device__ GEC_INLINE static void
    sample_inclusive(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT upper,
                     GecRng<Rng> &rng) {
        sample_inclusive_raw(a, upper.array(), rng);
    }
    template <typename Rng, typename Ctx>
    __host__ __device__ GEC_INLINE static void
    sample_inclusive(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT lower,
                     const Core &GEC_RSTRCT upper, GecRng<Rng> &rng, Ctx &ctx) {
        auto &ctx_view = ctx.template view_as<Core>();
        auto &span = ctx_view.template get<0>();
        Core::sub(span, upper, lower);
        sample_inclusive(a, span, rng);
        Core::add(a, lower);
    }

    template <typename Rng>
    __host__ __device__ static void
    sample_inclusive_raw(Core &GEC_RSTRCT a, const LIMB_T *GEC_RSTRCT bound,
                         GecRng<Rng> &rng) {
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

                a.array()[k] = rng.sample(mask);
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
                         GecRng<Rng> &rng) {
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

                a.array()[k] = rng.sample(mask);
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
                                                    GecRng<Rng> &rng) {
        do {
            sample(a, rng);
        } while (a.is_zero());
    }
};

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_MIXIN_RANDOM_HPP
