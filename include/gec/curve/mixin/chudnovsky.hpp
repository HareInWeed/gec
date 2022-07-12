#pragma once
#ifndef GEC_CURVE_MIXIN_CHUDNOVSKY_HPP
#define GEC_CURVE_MIXIN_CHUDNOVSKY_HPP

#include <gec/utils/crtp.hpp>

namespace gec {

namespace curve {

/** @brief mixin that enables ...
 */
template <typename Core, typename FIELD_T, const FIELD_T &A, const FIELD_T &B>
class GEC_EMPTY_BASES Chudnovsky
    : protected CRTP<Core, Chudnovsky<Core, FIELD_T, A, B>> {
    friend CRTP<Core, Chudnovsky<Core, FIELD_T, A, B>>;

  public:
    __host__ __device__ GEC_INLINE bool is_inf() {
        // TODO
    }
    __host__ __device__ GEC_INLINE void set_inf() {
        // TODO
    }

    __host__ __device__ GEC_INLINE static bool eq(const Core &GEC_RSTRCT a,
                                                  const Core &GEC_RSTRCT b) {
        // TODO
    }

    __host__ __device__ GEC_INLINE static bool
    on_curve(const Core &GEC_RSTRCT a) {
        // TODO
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add_distinct(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b,
                 const Core &GEC_RSTRCT c, F_CTX &GEC_RSTRCT ctx) {
        //  TODO
    }

    template <typename F_CTX>
    __host__ __device__ static void add_self(Core &GEC_RSTRCT a,
                                             const Core &GEC_RSTRCT b,
                                             F_CTX &GEC_RSTRCT ctx) {
        //  TODO
    }

    template <typename F_CTX>
    __host__ __device__ static void
    add(Core &GEC_RSTRCT a, const Core &GEC_RSTRCT b, const Core &GEC_RSTRCT c,
        F_CTX &GEC_RSTRCT ctx) {
        //  TODO
    }

    __host__ __device__ GEC_INLINE static void neg(Core &GEC_RSTRCT a,
                                                   const Core &GEC_RSTRCT b) {
        // TODO
    }
};

} // namespace curve

} // namespace gec

#endif // !GEC_CURVE_MIXIN_CHUDNOVSKY_HPP
