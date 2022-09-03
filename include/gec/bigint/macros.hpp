#pragma once
#ifndef GEC_BIGINT_MACROS_HPP
#define GEC_BIGINT_MACROS_HPP

// ---------- AddGroup ----------

#ifdef __CUDACC__
#define GEC_RAW_ADD_GROUP(T, N, MOD)                                           \
    ::gec::bigint::RawAddGroup<T, N, &MOD, &d_##MOD>
#define GEC_ALIGNED_RAW_ADD_GROUP(T, N, align, MOD)                            \
    ::gec::bigint::RawAddGroup<T, N, &MOD, &d_##MOD, align>
#define GEC_BASE_ADD_GROUP(Base, MOD)                                          \
    ::gec::bigint::BaseAddGroup<Base, &MOD, &d_##MOD>
#define GEC_ADD_GROUP(T, N, MOD) ::gec::bigint::AddGroup<T, N, &MOD, &d_##MOD>
#define GEC_ALIGNED_ADD_GROUP(T, N, align, MOD)                                \
    ::gec::bigint::AddGroup<T, N, &MOD, &d_##MOD, align>
#else
#define GEC_RAW_ADD_GROUP(T, N, MOD)                                           \
    ::gec::bigint::RawAddGroup<T, N, &MOD, nullptr>
#define GEC_ALIGNED_RAW_ADD_GROUP(T, N, align, MOD)                            \
    ::gec::bigint::RawAddGroup<T, N, &MOD, nullptr, align>
#define GEC_BASE_ADD_GROUP(Base, MOD)                                          \
    ::gec::bigint::BaseAddGroup<Base, &MOD, nullptr>
#define GEC_ADD_GROUP(T, N, MOD) ::gec::bigint::AddGroup<T, N, &MOD, nullptr>
#define GEC_ALIGNED_ADD_GROUP(T, N, align, MOD)                                \
    ::gec::bigint::AddGroup<T, N, &MOD, nullptr, align>
#endif

// ---------- Field ----------

#ifdef __CUDACC__
#define GEC_RAW_FIELD(T, N, MOD, MOD_P, RR, ONE_R)                             \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD, &d_##RR, \
                            &d_##ONE_R>
#define GEC_ALIGNED_RAW_FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)              \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD, &d_##RR, \
                            &d_##ONE_R, align>
#define GEC_BASE_FIELD(Base, MOD, MOD_P, RR, ONE_R)                            \
    ::gec::bigint::BaseField<Base, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD,         \
                             &d_##RR, &d_##ONE_R>
#define GEC_FIELD(T, N, MOD, MOD_P, RR, ONE_R)                                 \
    ::gec::bigint::Field<T, N, &MOD, MOD_P, &RR, &ONE_R, &d_##MOD, &d_##RR,    \
                         &d_##ONE_R>
#define GEC_ALIGNED_FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                  \
    ::gec::bigint::AlignedField<T, N, align, &MOD, MOD_P, &RR, &ONE_R,         \
                                &d_##MOD, &d_##RR, &d_##ONE_R>
#else
#define GEC_RAW_FIELD(T, N, MOD, MOD_P, RR, ONE_R)                             \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr,  \
                            nullptr>
#define GEC_ALIGNED_RAW_FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)              \
    ::gec::bigint::RawField<T, N, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr,  \
                            nullptr, align>
#define GEC_BASE_FIELD(Base, MOD, MOD_P, RR, ONE_R)                            \
    ::gec::bigint::BaseField<Base, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr, \
                             nullptr>
#define GEC_FIELD(T, N, MOD, MOD_P, RR, ONE_R)                                 \
    ::gec::bigint::Field<T, N, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr,     \
                         nullptr>
#define GEC_ALIGNED_FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                  \
    ::gec::bigint::Field<T, N, &MOD, MOD_P, &RR, &ONE_R, nullptr, nullptr,     \
                         nullptr, align>
#endif

#ifdef GEC_ENABLE_AVX2

#define GEC_RAW_AVX2FIELD(T, N, MOD, MOD_P, RR, ONE_R)                         \
    ::gec::bigint::RawAVX2Field<T, N, &MOD, MOD_P, &RR, &ONE_R>
#define GEC_ALIGNED_RAW_AVX2FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)          \
    ::gec::bigint::RawAVX2Field<T, N, &MOD, MOD_P, &RR, &ONE_R, align>
#define GEC_BASE_AVX2FIELD(Base, MOD, MOD_P, RR, ONE_R)                        \
    ::gec::bigint::BaseAVX2Field<Base, &MOD, MOD_P, &RR, &ONE_R>
#define GEC_AVX2FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)                      \
    ::gec::bigint::AVX2Field<T, N, &MOD, MOD_P, &RR, &ONE_R>
#define GEC_ALIGNED_AVX2FIELD(T, N, align, MOD, MOD_P, RR, ONE_R)              \
    ::gec::bigint::AVX2Field<T, N, &MOD, MOD_P, &RR, &ONE_R, align>

#endif // GEC_ENABLE_AVX2

#endif // !GEC_BIGINT_MACROS_HPP
