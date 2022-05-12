#pragma once
#ifndef GEC_CONTEXT_CHECK_HPP
#define GEC_CONTEXT_CHECK_HPP

namespace gec {

namespace utils {

#define GEC_CTX_CAP(ctx, cap)                                                  \
    static_assert((ctx::capacity) >= (cap),                                    \
                  "the capacity of context must be at least " #cap);

} // namespace utils

} // namespace gec

#endif // !GEC_CONTEXT_CHECK_HPP