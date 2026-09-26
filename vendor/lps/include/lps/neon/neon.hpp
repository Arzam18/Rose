#pragma once

#include "lps/neon/basic_vector_mask.hpp"
#include "lps/neon/neon.fwd.hpp"
#include "lps/neon/vector.hpp"
#include "lps/stdint.hpp"

namespace lps::neon {

  // ARM64 NEON provides a 128-bit native SIMD register width.
  struct environment {
    template<class T, usize N>
    using vector = neon::vector<T, N>;

    template<class T, usize N>
    using vector_mask = neon::vector_mask<T, N>;

    template<class T, usize N>
    using mask = neon::vector_mask<T, N>;

    template<class T>
    inline static constexpr usize native_vector_size = 16 * sizeof(u8) / sizeof(T);
  };

}  // namespace lps::neon
