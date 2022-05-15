#pragma once
#ifndef GEC_CONTEXT_HPP
#define GEC_CONTEXT_HPP

#include <gec/utils/basic.hpp>

#include <type_traits>

namespace gec {

namespace utils {

template <typename T, size_t N>
class Context : public Context<T, N - 1> {
  public:
    const static size_t capacity = N;

    T data;

    __host__ __device__ GEC_INLINE Context() : data(), Context<T, N - 1>() {}
    __host__ __device__ GEC_INLINE Context(const Context &other)
        : data(other.data), Context<T, N - 1>(
                                static_cast<const Context<T, N - 1> &>(other)) {
    }

    template <size_t M, std::enable_if_t<(M < N)> * = nullptr>
    __host__ __device__ GEC_INLINE Context(const Context<T, M> &other)
        : Context<T, N - 1>(other) {}

    template <typename U, typename... Args>
    __host__ __device__ GEC_INLINE Context(const U &data, const Args &...args)
        : data(data), Context<T, N - 1>(args...) {}
    template <typename U, typename... Args>
    __host__ __device__ GEC_INLINE Context(U &data, Args &...args)
        : data(data), Context<T, N - 1>(args...) {}

    /** @brief get the `I`th element in the context
     */
    template <size_t I, std::enable_if_t<(I < N)> * = nullptr>
    __host__ __device__ GEC_INLINE T &get() {
        return static_cast<Context<T, N - I> *>(this)->data;
    }
    /** @brief get the `I`th element in the context
     */
    template <size_t I, std::enable_if_t<(I < N)> * = nullptr>
    __host__ __device__ GEC_INLINE const T &get() const {
        return static_cast<const Context<T, N - I> *>(this)->data;
    }

    /** @brief remaining context after taking `I` elements
     */
    template <size_t I, std::enable_if_t<(0 < I && I <= N)> * = nullptr>
    __host__ __device__ GEC_INLINE Context<T, N - I> &rest() {
        return *static_cast<Context<T, N - I> *>(this);
    }
    /** @brief remaining context after taking `I` elements
     */
    template <size_t I, std::enable_if_t<(0 < I && I <= N)> * = nullptr>
    __host__ __device__ GEC_INLINE const Context<T, N - I> &rest() const {
        return *static_cast<const Context<T, N - I> *>(this);
    }
};

template <typename T>
class Context<T, 0> {};

} // namespace utils

} // namespace gec

#endif // !GEC_CONTEXT_HPP