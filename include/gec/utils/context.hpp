#pragma once
#ifndef GEC_CONTEXT_HPP
#define GEC_CONTEXT_HPP

#include <gec/utils/basic.hpp>

#include <type_traits>

namespace gec {

namespace utils {

template <size_t I, typename... Types>
struct CtxTypeI {};
template <size_t I, typename T, typename... Types>
struct CtxTypeI<I, T, Types...> {
    using type = typename CtxTypeI<I - 1, Types...>::type;
};
template <typename T, typename... Types>
struct CtxTypeI<0, T, Types...> {
    using type = T;
};

template <size_t I, size_t align>
struct AlignTo {
    static const size_t value = align * ((I + align - 1) / align);
};

template <size_t align, typename T, typename... Types>
struct CtxAlignChecker {
    __host__ __device__ constexpr static bool call() {
        return alignof(T) <= align && CtxAlignChecker<align, Types...>::call();
    }
};
template <size_t align, typename T>
struct CtxAlignChecker<align, T> {
    __host__ __device__ constexpr static bool call() {
        return alignof(T) <= align;
    }
};

template <size_t I, size_t occupied, typename... Types>
struct CtxGetOffset;
template <size_t I, size_t occupied, typename T, typename... Types>
struct CtxGetOffset<I, occupied, T, Types...> {
    __host__ __device__ constexpr static size_t call() {
        return CtxGetOffset<I - 1,
                            AlignTo<occupied, alignof(T)>::value + sizeof(T),
                            Types...>::call();
    }
};
template <size_t occupied, typename T, typename... Types>
struct CtxGetOffset<0, occupied, T, Types...> {
    __host__ __device__ constexpr static size_t call() {
        return AlignTo<occupied, alignof(T)>::value;
    }
};
template <size_t occupied>
struct CtxGetOffset<0, occupied> {
    __host__ __device__ constexpr static size_t call() { return occupied; }
};

template <size_t I, size_t occupied, typename... Types>
struct CtxGetEndOffset;
template <size_t I, size_t occupied, typename T, typename... Types>
struct CtxGetEndOffset<I, occupied, T, Types...> {
    __host__ __device__ constexpr static size_t call() {
        return CtxGetOffset<I - 1, occupied, T, Types...>::call() +
               sizeof(typename CtxTypeI<I - 1, T, Types...>::type);
    }
};
template <size_t occupied>
struct CtxGetEndOffset<0, occupied> {
    __host__ __device__ constexpr static size_t call() { return occupied; }
};

template <size_t N, size_t align, size_t occupied, typename... Types>
class GEC_EMPTY_BASES alignas(align) Context {
  public:
    static const size_t capacity = N - occupied;
    uint8_t mem[N];

    __host__ __device__ GEC_INLINE constexpr Context() {}

    template <size_t I>
    __host__ __device__ GEC_INLINE static constexpr size_t get_offset() {
        return CtxGetOffset<I, occupied, Types...>::call();
    }

    template <size_t I, std::enable_if_t<(I < sizeof...(Types))> * = nullptr>
    __host__ __device__ GEC_INLINE constexpr
        typename CtxTypeI<I, Types...>::type &
        get() {
        return *reinterpret_cast<typename CtxTypeI<I, Types...>::type *>(
            this->mem + get_offset<I>());
    }
    template <size_t I, std::enable_if_t<(I < sizeof...(Types))> * = nullptr>
    __host__ __device__
        GEC_INLINE constexpr const typename CtxTypeI<I, Types...>::type &
        get() const {
        return *reinterpret_cast<typename CtxTypeI<I, Types...>::type *>(
            this->mem + get_offset<I>());
    }

    __host__ __device__ GEC_INLINE constexpr Context<
        N, align, CtxGetEndOffset<sizeof...(Types), occupied, Types...>::call()>
        &rest() {
        return *reinterpret_cast<Context<
            N, align,
            CtxGetEndOffset<sizeof...(Types), occupied, Types...>::call()> *>(
            this);
    }
    __host__ __device__ GEC_INLINE constexpr const Context<
        N, align, CtxGetEndOffset<sizeof...(Types), occupied, Types...>::call()>
        &rest() const {
        return *reinterpret_cast<const Context<
            N, align,
            CtxGetEndOffset<sizeof...(Types), occupied, Types...>::call()> *>(
            this);
    }

    template <typename... OtherTypes>
    __host__ __device__
        GEC_INLINE constexpr Context<N, align, occupied, OtherTypes...> &
        view_as() {
        static_assert(CtxAlignChecker<align, OtherTypes...>::call(),
                      "alignments of some components exceed context limit");
        static_assert(
            CtxGetEndOffset<sizeof...(OtherTypes), occupied,
                            OtherTypes...>::call() <= N,
            "context does not have enough space to hold all components");
        return *reinterpret_cast<Context<N, align, occupied, OtherTypes...> *>(
            this);
    }
    template <typename... OtherTypes>
    __host__ __device__
        GEC_INLINE constexpr const Context<N, align, occupied, OtherTypes...> &
        view_as() const {
        static_assert(CtxAlignChecker<align, OtherTypes...>::call(),
                      "alignments of some components exceed context limit");
        static_assert(
            CtxGetEndOffset<sizeof...(OtherTypes), occupied,
                            OtherTypes...>::call() <= N,
            "context does not have enough space to hold all components");
        return *reinterpret_cast<
            const Context<N, align, occupied, OtherTypes...> *>(this);
    }
};

} // namespace utils

} // namespace gec

#endif // !GEC_CONTEXT_HPP