#pragma once
#ifndef GEC_BIGINT_DATA_LITERAL_HPP
#define GEC_BIGINT_DATA_LITERAL_HPP

#include <gec/utils/basic.hpp>
#include <type_traits>

namespace gec {

namespace bigint {

namespace literal {

// basic architecture:
// raw literal --> IntLiteral --> ByteCons --> ConstArray

// ----- ConstArray -----

template <typename T, T... comps>
struct ConstArray {
    static constexpr T arr[sizeof...(comps)] = {comps...};
    static constexpr const T &get(size_t i) { return arr[i]; }
};

// ----- Cons -----

struct Nil {};
template <typename T, T First, typename Rest>
struct Cons {};

namespace _literal_ {

constexpr size_t byte_bits = 8;

// ----- ConstArray & Cons conversion -----

/// @brief convert ConstArray into Cons
template <typename A>
struct ToCons;
template <typename A>
using ToCons_t = typename ToCons<A>::type;
template <typename T>
struct ToCons<ConstArray<T>> {
    using type = Nil;
};
template <typename T, T c, T... comps>
struct ToCons<ConstArray<T, c, comps...>> {
    using type = Cons<T, c, ToCons_t<ConstArray<T, comps...>>>;
};

/// @brief convert Cons into ConstArray
template <typename A>
struct ToConstArray;
template <typename A>
using ToConstArray_t = typename ToConstArray<A>::type;
template <typename A, typename T, T... comps>
struct ToConstArrayHelper;
template <typename A, typename T, T... comps>
using Cons2ConstArrayHelper_t =
    typename ToConstArrayHelper<A, T, comps...>::type;

template <typename T, T First, typename Rest>
struct ToConstArray<Cons<T, First, Rest>> {
    using type = Cons2ConstArrayHelper_t<Cons<T, First, Rest>, T>;
};
template <typename T, T... comps>
struct ToConstArrayHelper<Nil, T, comps...> {
    using type = ConstArray<T, comps...>;
};
template <typename T, T c, typename Rest, T... comps>
struct ToConstArrayHelper<Cons<T, c, Rest>, T, comps...> {
    using type = Cons2ConstArrayHelper_t<Rest, T, comps..., c>;
};

// ----- Len -----

template <typename C>
struct Len;
template <typename T, T... comps>
struct Len<ConstArray<T, comps...>> {
    constexpr static size_t value = sizeof...(comps);
};
template <typename T, T First, typename Rest>
struct Len<Cons<T, First, Rest>> {
    constexpr static size_t value = 1 + Len<Rest>::value;
};
template <>
struct Len<Nil> {
    constexpr static size_t value = 0;
};

// ----- Car & Cdr -----

template <typename C>
struct Car;
template <typename T, T First, typename Rest>
struct Car<Cons<T, First, Rest>> {
    static constexpr T value = First;
};
template <typename T, T c, T... comps>
struct Car<ConstArray<T, c, comps...>> {
    static constexpr T value = c;
};

template <typename C>
struct Cdr;
template <typename C>
using Cdr_t = typename Cdr<C>::type;
template <typename T, T First, typename Rest>
struct Cdr<Cons<T, First, Rest>> {
    using type = Rest;
};
template <typename T, T c, T... comps>
struct Cdr<ConstArray<T, c, comps...>> {
    using type = ConstArray<T, comps...>;
};

// ----- Append -----

template <typename AC, typename T, T elem>
struct Append;
template <typename AC, typename T, T elem>
using Append_t = typename Append<AC, T, elem>::type;
template <typename T, T elem, T... comps>
struct Append<ConstArray<T, comps...>, T, elem> {
    using type = ConstArray<T, comps..., elem>;
};

// ----- Indexer -----

template <typename C, size_t i>
struct Indexer;
template <typename T, T First, typename Rest, size_t i>
struct Indexer<Cons<T, First, Rest>, i> {
    static constexpr T value = Indexer<Rest, i - 1>::value;
};
template <typename T, T First, typename Rest>
struct Indexer<Cons<T, First, Rest>, 0> {
    static constexpr T value = First;
};
template <typename T, T c, size_t i, T... comps>
struct Indexer<ConstArray<T, c, comps...>, i> {
    static constexpr T value = Indexer<ConstArray<T, comps...>, i - 1>::value;
};
template <typename T, T c, T... comps>
struct Indexer<ConstArray<T, c, comps...>, 0> {
    static constexpr T value = c;
};

// ----- Reverse -----

template <typename C>
struct Reverse;
template <typename C>
using Reverse_t = typename Reverse<C>::type;
template <typename C, typename R>
struct ReverseHelper;
template <typename C, typename R>
using ReverseHelper_t = typename ReverseHelper<C, R>::type;

template <>
struct Reverse<Nil> {
    using type = Nil;
};
template <typename T, T First, typename Rest>
struct Reverse<Cons<T, First, Rest>> {
    using type = ReverseHelper_t<Cons<T, First, Rest>, Nil>;
};
template <typename T, T... comps>
struct Reverse<ConstArray<T, comps...>> {
    using type = ToConstArray_t<Reverse_t<ToCons_t<ConstArray<T, comps...>>>>;
};
template <typename R>
struct ReverseHelper<Nil, R> {
    using type = R;
};
template <typename T, T First, typename Rest, typename R>
struct ReverseHelper<Cons<T, First, Rest>, R> {
    using type = ReverseHelper_t<Rest, Cons<T, First, R>>;
};

// ----- RoundLeft -----

template <typename DL, size_t i>
struct PadLeftZeros;
template <typename DL, size_t i>
using PadLeftZeros_t = typename PadLeftZeros<DL, i>::type;

template <typename T, T First, typename Rest, size_t i>
struct PadLeftZeros<Cons<T, First, Rest>, i> {
    using type = PadLeftZeros_t<Cons<T, 0, Cons<T, First, Rest>>, i - 1>;
};
template <typename T, T First, typename Rest>
struct PadLeftZeros<Cons<T, First, Rest>, 0> {
    using type = Cons<T, First, Rest>;
};

template <typename T, size_t i, T... comps>
struct PadLeftZerosCAHelper;
template <typename T, size_t i, T... comps>
using PadLeftZerosCAHelper_t =
    typename PadLeftZerosCAHelper<T, i, comps...>::type;
template <typename T, size_t i, T... comps>
struct PadLeftZerosCAHelper<ConstArray<T, comps...>, i> {
    using type = PadLeftZerosCAHelper_t<T, i - 1, 0, comps...>;
};
template <typename T, T... comps>
struct PadLeftZerosCAHelper<ConstArray<T, comps...>, 0> {
    using type = ConstArray<T, comps...>;
};

template <typename DL, size_t i>
struct PadRightZeros;
template <typename DL, size_t i>
using PadRightZeros_t = typename PadRightZeros<DL, i>::type;

template <typename T, size_t i, T... comps>
struct PadRightZerosCAHelper;
template <typename T, size_t i, T... comps>
using PadRightZerosCAHelper_t =
    typename PadRightZerosCAHelper<T, i, comps...>::type;
template <typename T, size_t i, T... comps>
struct PadRightZerosCAHelper<ConstArray<T, comps...>, i> {
    using type = PadRightZerosCAHelper_t<T, i - 1, comps..., 0>;
};
template <typename T, T... comps>
struct PadRightZerosCAHelper<ConstArray<T, comps...>, 0> {
    using type = ConstArray<T, comps...>;
};

constexpr size_t padding_value(size_t len, size_t group) {
    return ((len + group - 1) / group) * group - len;
}

template <typename DL, size_t group>
using RoundLeft_t =
    typename PadLeftZeros<DL, padding_value(Len<DL>::value, group)>::type;

template <typename DL, size_t group>
using RoundRight_t =
    typename PadRightZeros<DL, padding_value(Len<DL>::value, group)>::type;

// ----- Literals -----

template <unsigned Base, typename DL>
struct IntLiteral {
    constexpr static unsigned base = Base;
    using Digits = DL;
};

// ----- Half -----

template <typename DL>
struct RemoveHighZeros;
template <typename DL>
using RemoveHighZeros_t = typename RemoveHighZeros<DL>::type;
template <typename DL>
struct RemoveHighZeros {
    using type = DL;
};
template <typename T, typename Rest>
struct RemoveHighZeros<Cons<T, 0, Rest>> {
    using type = RemoveHighZeros_t<Rest>;
};
template <typename T>
struct RemoveHighZeros<Cons<T, 0, Nil>> {
    using type = Cons<T, 0, Nil>;
};

template <typename IL>
class Half;
template <unsigned base, typename DL, unsigned r>
class HalfHelper;

template <unsigned base, typename DL>
class Half<IntLiteral<base, DL>> {
    using Helper = HalfHelper<base, DL, 0>;

  public:
    using q = IntLiteral<base, RemoveHighZeros_t<typename Helper::q>>;
    static constexpr unsigned r = Helper::r;
};
template <unsigned base, typename T, T First, typename Rest, unsigned R>
class HalfHelper<base, Cons<T, First, Rest>, R> {
    using Next = HalfHelper<base, Rest, (First % 2) * (base / 2)>;

  public:
    using q = Cons<T, First / 2 + R, typename Next::q>;
    static constexpr unsigned r = Next::r;
};
template <unsigned base, typename T, T First, unsigned R>
class HalfHelper<base, Cons<T, First, Nil>, R> {
  public:
    using q = Cons<T, First / 2 + R, Nil>;
    static constexpr unsigned r = First % 2;
};

} // namespace _literal_

// ----- TakeExact -----

namespace _literal_ {

template <typename R, typename AC, size_t len>
struct TakeExactHelper;
template <typename R, typename AC, size_t len>
using TakeExactHelper_t = typename TakeExactHelper<R, AC, len>::type;

} // namespace _literal_

template <typename AC, size_t len>
struct TakeExact;
template <typename AC, size_t len>
using TakeExact_t = typename TakeExact<AC, len>::type;
template <typename T, size_t len, T... comps>
struct TakeExact<ConstArray<T, comps...>, len> {
    using type = _literal_::TakeExactHelper_t<ConstArray<T>,
                                              ConstArray<T, comps...>, len>;
};

namespace _literal_ {

template <typename R, typename T, size_t len, T c, T... comps>
struct TakeExactHelper<R, ConstArray<T, c, comps...>, len> {
    using type =
        TakeExactHelper_t<Append_t<R, T, c>, ConstArray<T, comps...>, len - 1>;
};
template <typename R, typename T, size_t len>
struct TakeExactHelper<R, ConstArray<T>, len> {
    using type = TakeExactHelper_t<Append_t<R, T, 0>, ConstArray<T>, len - 1>;
};
template <typename R, typename T, T... comps>
struct TakeExactHelper<R, ConstArray<T, comps...>, 0> {
    using type = R;
};
template <typename R, typename T>
struct TakeExactHelper<R, ConstArray<T>, 0> {
    using type = R;
};

} // namespace _literal_

// ----- ToLimbArray -----

namespace _literal_ {

template <typename BC, typename LimbT, LimbT... comps>
struct ToLimbArrayHelper;
template <typename BC, typename LimbT, LimbT... comps>
using ToLimbArrayHelper_t =
    typename ToLimbArrayHelper<BC, LimbT, comps...>::type;

template <typename LimbT, LimbT... comps>
struct ToLimbArrayHelper<Nil, LimbT, comps...> {
    using type = ConstArray<LimbT, comps...>;
};
template <typename BC>
struct ToLimbArrayHelper<BC, uint8_t> {
    using type = ToConstArray_t<BC>;
};
template <unsigned p0, unsigned p1, typename Rest, uint16_t... comps>
struct ToLimbArrayHelper<Cons<unsigned, p0, Cons<unsigned, p1, Rest>>, uint16_t,
                         comps...> {
    using type = ToLimbArrayHelper_t<Rest, uint16_t, comps...,
                                     uint16_t(p0) | (uint16_t(p1) << 8)>;
};
template <unsigned p0, unsigned p1, unsigned p2, unsigned p3, typename Rest,
          uint32_t... comps>
struct ToLimbArrayHelper<
    Cons<unsigned, p0,
         Cons<unsigned, p1, Cons<unsigned, p2, Cons<unsigned, p3, Rest>>>>,
    uint32_t, comps...> {
    using type =
        ToLimbArrayHelper_t<Rest, uint32_t, comps...,
                            uint32_t(p0) | (uint32_t(p1) << 8) |
                                (uint32_t(p2) << 16) | (uint32_t(p3) << 24)>;
};
template <unsigned p0, unsigned p1, unsigned p2, unsigned p3, unsigned p4,
          unsigned p5, unsigned p6, unsigned p7, typename Rest,
          uint64_t... comps>
struct ToLimbArrayHelper<
    Cons<unsigned, p0,
         Cons<unsigned, p1,
              Cons<unsigned, p2,
                   Cons<unsigned, p3,
                        Cons<unsigned, p4,
                             Cons<unsigned, p5,
                                  Cons<unsigned, p6,
                                       Cons<unsigned, p7, Rest>>>>>>>>,
    uint64_t, comps...> {
    using type =
        ToLimbArrayHelper_t<Rest, uint64_t, comps...,
                            uint64_t(p0) | (uint64_t(p1) << 8) |
                                (uint64_t(p2) << 16) | (uint64_t(p3) << 24) |
                                (uint64_t(p4) << 32) | (uint64_t(p5) << 40) |
                                (uint64_t(p6) << 48) | (uint64_t(p7) << 56)>;
};

} // namespace _literal_

template <typename BC, typename LimbT>
struct ToLimbArray;
template <typename BC, typename LimbT>
using ToLimbArray_t = typename ToLimbArray<BC, LimbT>::type;

template <typename BC, typename LimbT>
struct ToLimbArray {
    using type = _literal_::ToLimbArrayHelper_t<
        _literal_::Reverse_t<_literal_::RoundLeft_t<
            BC, utils::type_bits<LimbT>::value / _literal_::byte_bits>>,
        LimbT>;
};

namespace _literal_ {

// ----- ToByteArray -----

template <typename IL>
class DrainByte;
template <size_t i, unsigned limb, typename IL>
class DrainByteHelper;

template <typename IL>
class DrainByte {
    using Helper = DrainByteHelper<0, 0, IL>;

  public:
    using rest = typename Helper::rest;
    static constexpr unsigned limb = Helper::limb;
};

template <size_t i, unsigned Limb, typename IL>
class DrainByteHelper {
    using HI = Half<IL>;
    using Next = DrainByteHelper<i + 1, (Limb | (HI::r << i)), typename HI::q>;

  public:
    using rest = typename Next::rest;
    static constexpr unsigned limb = Next::limb;
};
template <unsigned Limb, typename IL>
class DrainByteHelper<byte_bits, Limb, IL> {
  public:
    using rest = IL;
    static constexpr unsigned limb = Limb;
};

template <typename IL>
struct ToByteCons;
template <typename IL>
using ToByteCons_t = typename ToByteCons<IL>::type;
template <typename IL, typename R>
class ToByteConsHelper;
template <typename IL, typename R>
using ToByteConsHelper_t = typename ToByteConsHelper<IL, R>::type;

template <typename IL>
struct ToByteCons {
    using type = ToByteConsHelper_t<IL, Nil>;
};
template <unsigned Base>
struct ToByteCons<IntLiteral<Base, Cons<unsigned, 0, Nil>>> {
    using type = Cons<unsigned, 0, Nil>;
};

template <typename IL, typename R>
class ToByteConsHelper {
    using Drained = DrainByte<IL>;

  public:
    using type = ToByteConsHelper_t<typename Drained::rest,
                                    Cons<unsigned, Drained::limb, R>>;
};
template <unsigned Base, typename R>
class ToByteConsHelper<IntLiteral<Base, Cons<unsigned, 0, Nil>>, R> {
  public:
    using type = R;
};

// specialized ToByteCons for binary and hexadecimal

template <typename DL>
struct ToByteConsBinHelper;
template <typename DL>
using ToByteConsBinHelper_t = typename ToByteConsBinHelper<DL>::type;
template <unsigned b0, unsigned b1, unsigned b2, unsigned b3, unsigned b4,
          unsigned b5, unsigned b6, unsigned b7, typename Rest>
struct ToByteConsBinHelper<
    Cons<unsigned, b7,
         Cons<unsigned, b6,
              Cons<unsigned, b5,
                   Cons<unsigned, b4,
                        Cons<unsigned, b3,
                             Cons<unsigned, b2,
                                  Cons<unsigned, b1,
                                       Cons<unsigned, b0, Rest>>>>>>>>> {
    using type = Cons<unsigned,
                      b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) |
                          (b5 << 5) | (b6 << 6) | (b7 << 7),
                      ToByteConsBinHelper_t<Rest>>;
};
template <>
struct ToByteConsBinHelper<Nil> {
    using type = Nil;
};

template <typename DL>
struct ToByteCons<IntLiteral<2, DL>> {
    using type = ToByteConsBinHelper_t<RoundLeft_t<DL, 8>>;
};
template <>
struct ToByteCons<IntLiteral<2, Cons<unsigned, 0, Nil>>> {
    using type = Cons<unsigned, 0, Nil>;
};

template <typename DL>
struct ToByteConsHexHelper;
template <typename DL>
using ToByteConsHexHelper_t = typename ToByteConsHexHelper<DL>::type;
template <unsigned h0, unsigned h1, typename Rest>
struct ToByteConsHexHelper<Cons<unsigned, h1, Cons<unsigned, h0, Rest>>> {
    using type = Cons<unsigned, h0 | (h1 << 4), ToByteConsHexHelper_t<Rest>>;
};
template <>
struct ToByteConsHexHelper<Nil> {
    using type = Nil;
};

template <typename DL>
struct ToByteCons<IntLiteral<16, DL>> {
    using type = ToByteConsHexHelper_t<RoundLeft_t<DL, 2>>;
};
template <>
struct ToByteCons<IntLiteral<16, Cons<unsigned, 0, Nil>>> {
    using type = Cons<unsigned, 0, Nil>;
};

// ----- literal operator -----

template <unsigned base, char c, typename Enable = void>
struct IsDigit {
    static constexpr bool value = false;
};
template <unsigned base, char c>
struct ToDigit {
    static constexpr unsigned value = unsigned(c - '0');
};

// binary
template <char c>
struct IsDigit<2, c, std::enable_if_t<(c == '0' || c == '1')>> {
    static constexpr bool value = true;
};

// octal
template <char c>
struct IsDigit<8, c, std::enable_if_t<(c >= '0' && c <= '7')>> {
    static constexpr bool value = true;
};

// decimal
template <char c>
struct IsDigit<10, c, std::enable_if_t<(c >= '0' && c <= '9')>> {
    static constexpr bool value = true;
};

// hexadecimal
template <char c>
struct IsDigit<
    16, c,
    std::enable_if_t<((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                      (c >= 'A' && c <= 'F'))>> {
    static constexpr bool value = true;
};
template <char c>
struct ToDigit<16, c> {
    static constexpr unsigned value =
        (c >= '0' && c <= '9'
             ? unsigned(c - '0')
             : (c >= 'a' && c <= 'f' ? unsigned(c - 'a' + 10)
                                     : unsigned(c - 'A' + 10)));
};

template <unsigned base, char... chars>
struct ToDigitLiteral;
template <unsigned base, char... chars>
using ToDigitLiteral_t = typename ToDigitLiteral<base, chars...>::type;
template <unsigned base>
struct ToDigitLiteral<base> {
    using type = Nil;
};
template <unsigned base, char c, char... chars>
struct ToDigitLiteral<base, c, chars...> {
    using type =
        typename std::conditional<IsDigit<base, c>::value,
                                  Cons<unsigned, ToDigit<base, c>::value,
                                       ToDigitLiteral_t<base, chars...>>,
                                  ToDigitLiteral_t<base, chars...>>::type;
};

template <char... chars>
struct ToIntLiteral;
template <char... chars>
using ToIntLiteral_t = typename ToIntLiteral<chars...>::type;
template <char... chars>
struct ToIntLiteral {
    using type = IntLiteral<10, ToDigitLiteral_t<10, chars...>>;
};
template <char... chars>
struct ToIntLiteral<'2', '.', chars...> {
    using type = IntLiteral<2, ToDigitLiteral_t<2, chars...>>;
};
template <char... chars>
struct ToIntLiteral<'8', '.', chars...> {
    using type = IntLiteral<8, ToDigitLiteral_t<8, chars...>>;
};
template <char... chars>
struct ToIntLiteral<'0', 'x', chars...> {
    using type = IntLiteral<16, ToDigitLiteral_t<16, chars...>>;
};

template <char... chars>
using ParseRawLiteral = ToByteCons_t<ToIntLiteral_t<chars...>>;

} // namespace _literal_

template <char... chars>
constexpr _literal_::ParseRawLiteral<chars...> operator"" _int() {
    return _literal_::ParseRawLiteral<chars...>();
}

} // namespace literal

} // namespace bigint

} // namespace gec

#endif // !GEC_BIGINT_DATA_LITERAL_HPP
