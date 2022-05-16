#pragma once
#ifndef GEC_CURVE_POINT_CONTEXT_HPP
#define GEC_CURVE_POINT_CONTEXT_HPP

#include <gec/utils/context.hpp>

namespace gec {

namespace curve {

// TODO: Maybe `PointContext` is useless once operations for heterogeneous data
// structures are implemented?

template <size_t CompN, size_t TN, size_t PN, size_t TI, size_t PI>
struct PointContextChecker {
    const static bool value =
        (TI <= TN ? PI <= PN : (TI - TN + CompN - 1) / CompN + PI <= PN);
};

template <typename P, size_t TN, size_t PN, size_t TI = 0, size_t PI = 0,
          std::enable_if_t<PointContextChecker<P::CompN, TN, PN, TI, PI>::value>
              * = nullptr>
class PointContext {
    // TODO: methods to create a reference point context? Or rather, methods to
    // create a reference point?
    using T = typename P::CompT;
    template <size_t N>
    using P_CTX = utils::Context<P, N>;
    template <size_t N>
    using T_CTX = utils::Context<T, N>;

  public:
    const static size_t capacity = TN + (PN - PI) * P::CompN - TI;
    const static size_t point_capacity =
        PN - PI - (TI <= TN ? 0 : (TI - TN + P::CompN - 1) / P::CompN);

    P_CTX<PN> ctx_p;
    T_CTX<TN> ctx_t;

    __host__ __device__ GEC_INLINE PointContext() : ctx_t(), ctx_p() {}
    __host__ __device__ GEC_INLINE PointContext(const PointContext &other)
        : ctx_t(other.ctx_t), ctx_p(other.ctx_p) {}

    template <size_t TM, size_t PM,
              std::enable_if_t<(TM <= TN && PM <= PN)> * = nullptr>
    __host__ __device__ GEC_INLINE PointContext(const T_CTX<TM> &ctx_t,
                                                const P_CTX<PM> &ctx_p)
        : ctx_t(ctx_t), ctx_p(ctx_p) {}

    /** @brief get the `I`th element of type `T` in the context
     */
    template <size_t I, std::enable_if_t<(TI + I < TN)> * = nullptr>
    __host__ __device__ GEC_INLINE T &get() {
        return ctx_t.template get<TI + I>();
    }
    /** @brief get the `I`th element of type `T` in the context
     */
    template <size_t I,
              std::enable_if_t<(TI + I >= TN && I < capacity)> * = nullptr>
    __host__ __device__ GEC_INLINE T &get() {
        constexpr size_t Idx = (TI + I) - TN;
        return ctx_p.template get<Idx / P::CompN>()
            .template get<Idx % P::CompN>();
    }

    /** @brief get the `I`th element of type `T` in the context
     */
    template <size_t I, std::enable_if_t<(TI + I < TN)> * = nullptr>
    __host__ __device__ GEC_INLINE const T &get() const {
        return ctx_t.template get<TI + I>();
    }
    /** @brief get the `I`th element of type `T` in the context
     */
    template <size_t I,
              std::enable_if_t<(TI + I >= TN && I < capacity)> * = nullptr>
    __host__ __device__ GEC_INLINE const T &get() const {
        constexpr size_t Idx = (TI + I) - TN;
        return ctx_p.template get<Idx / P::CompN>()
            .template get<Idx % P::CompN>();
    }

    /** @brief get the `I`th element of type `P` in the context
     */
    template <size_t I, std::enable_if_t<(I < point_capacity)> * = nullptr>
    __host__ __device__ GEC_INLINE P &get_p() {
        return ctx_p.template get<PN - 1 - (PI + I)>();
    }
    /** @brief get the `I`th element of type `P` in the context
     */
    template <size_t I, std::enable_if_t<(I < point_capacity)> * = nullptr>
    __host__ __device__ GEC_INLINE const P &get_p() const {
        return ctx_p.template get<PN - 1 - (PI + I)>();
    }

    /** @brief remaining context after taking `I` elements of `T` and `J`
     * elements of `P`
     */
    template <size_t I, size_t J = 0,
              std::enable_if_t<(
                  (I > 0 || J > 0) &&
                  PointContextChecker<P::CompN, TN, PN, TI + I, PI + J>::value)>
                  * = nullptr>
    __host__ __device__ GEC_INLINE PointContext<P, TN, PN, TI + I, PI + J> &
    rest() {
        return reinterpret_cast<PointContext<P, TN, PN, TI + I, PI + J> &>(
            *this);
    }
    /** @brief remaining context after taking `I` elements of `T` and `J`
     * elements of `P`
     */
    template <size_t I, size_t J = 0,
              std::enable_if_t<(
                  (I > 0 || J > 0) &&
                  PointContextChecker<P::CompN, TN, PN, TI + I, PI + J>::value)>
                  * = nullptr>
    __host__ __device__ GEC_INLINE const
        PointContext<P, TN, PN, TI + I, PI + J> &
        rest() const {
        return reinterpret_cast<PointContext<P, TN, PN, TI + I, PI + J> &>(
            *this);
    }
};

// TODO: a better `CompoundContext` that capture the minmum bound of elements
// and points?

} // namespace curve

} // namespace gec

#endif // GEC_CURVE_POINT_CONTEXT_HPP