#pragma once

#include "lps/detail/msb.hpp"
#include "lps/detail/vector_clamped_size.hpp"
#include "lps/neon/basic_vector_mask.def.hpp"
#include "lps/neon/vector.def.hpp"
#include <arm_neon.h>
#include "lps/stdint.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>

namespace lps::neon {

namespace neon_impl {

template<class T, usize N>
inline auto neon_load(const void* src) {
  if constexpr (std::is_same_v<T, i8> && N == 16) {
    return vreinterpretq_s8_u8(
        vld1q_u8(reinterpret_cast<const u8*>(src)));
  } else if constexpr (std::is_same_v<T, u8> && N == 16) {
    return vld1q_u8(reinterpret_cast<const u8*>(src));
  } else if constexpr (std::is_same_v<T, i16> && N == 8) {
    return vld1q_s16(reinterpret_cast<const i16*>(src));
  } else if constexpr (std::is_same_v<T, u16> && N == 8) {
    return vld1q_u16(reinterpret_cast<const u16*>(src));
  } else if constexpr (std::is_same_v<T, i32> && N == 4) {
    return vld1q_s32(reinterpret_cast<const i32*>(src));
  } else if constexpr (std::is_same_v<T, u32> && N == 4) {
    return vld1q_u32(reinterpret_cast<const u32*>(src));
  } else if constexpr (std::is_same_v<T, i64> && N == 2) {
    return vld1q_s64(reinterpret_cast<const i64*>(src));
  } else if constexpr (std::is_same_v<T, u64> && N == 2) {
    return vld1q_u64(reinterpret_cast<const u64*>(src));
  }
}

template<class T, usize N, class V>
inline void neon_store(void* dst, V v) {
  if constexpr (std::is_same_v<T, i8> && N == 16) {
    vst1q_s8(
        reinterpret_cast<i8*>(dst),
        vreinterpretq_s8_u8(v));
  } else if constexpr (std::is_same_v<T, u8> && N == 16) {
    vst1q_u8(
        reinterpret_cast<u8*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, i16> && N == 8) {
    vst1q_s16(
        reinterpret_cast<i16*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, u16> && N == 8) {
    vst1q_u16(
        reinterpret_cast<u16*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, i32> && N == 4) {
    vst1q_s32(
        reinterpret_cast<i32*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, u32> && N == 4) {
    vst1q_u32(
        reinterpret_cast<u32*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, i64> && N == 2) {
    vst1q_s64(
        reinterpret_cast<i64*>(dst),
        v);
  } else if constexpr (std::is_same_v<T, u64> && N == 2) {
    vst1q_u64(
        reinterpret_cast<u64*>(dst),
        v);
  }
}

} // namespace neon_impl


template<class T, usize N>
constexpr vector<T, N> vector<T, N>::zero() {
  return vector { std::array<T, N> {} };
}


template<class T, usize N>
constexpr vector<T, N> vector<T, N>::splat(T value) {
  vector v;
  std::fill(v.raw.begin(), v.raw.end(), value);
  return v;
}


template<class T, usize N>
constexpr vector<T, N> vector<T, N>::splat(half_vector value) {
  vector v;
  std::copy(value.raw.begin(), value.raw.end(), v.raw.begin());
  std::copy(
      value.raw.begin(),
      value.raw.end(),
      v.raw.begin() + N / 2);
  return v;
}


template<class T, usize N>
vector<T, N> vector<T, N>::load(const void* src) {
  vector v;

  if constexpr (
      (N * sizeof(T) == 16) &&
      (std::is_same_v<T, i8> ||
       std::is_same_v<T, u8> ||
       std::is_same_v<T, i16> ||
       std::is_same_v<T, u16> ||
       std::is_same_v<T, i32> ||
       std::is_same_v<T, u32> ||
       std::is_same_v<T, i64> ||
       std::is_same_v<T, u64>)) {

    auto value = neon_impl::neon_load<T, N>(src);
    neon_impl::neon_store<T, N>(v.raw.data(), value);

  } else {
    std::memcpy(
        v.raw.data(),
        src,
        sizeof(v.raw));
  }

  return v;
}


template<class T, usize N>
void vector<T, N>::store(void* dst) {
  if constexpr (
      (N * sizeof(T) == 16) &&
      (std::is_same_v<T, i8> ||
       std::is_same_v<T, u8> ||
       std::is_same_v<T, i16> ||
       std::is_same_v<T, u16> ||
       std::is_same_v<T, i32> ||
       std::is_same_v<T, u32> ||
       std::is_same_v<T, i64> ||
       std::is_same_v<T, u64>)) {

    auto value = neon_impl::neon_load<T, N>(raw.data());
    neon_impl::neon_store<T, N>(dst, value);

  } else {
    std::memcpy(
        dst,
        raw.data(),
        sizeof(raw));
  }
}


template<class T, usize N>
constexpr T vector<T, N>::read(usize i) const {
  return raw[i];
}


template<class T, usize N>
template<class U>
constexpr vector<U, detail::clamped_size<U, N>>
vector<T, N>::convert() const {

  vector<U, detail::clamped_size<U, N>> result {};

  for (usize i = 0; i < N; i++) {
    result.raw[i] = static_cast<U>(raw[i]);
  }

  return result;
}


template<class T, usize N>
template<class V, usize extract_index>
constexpr V vector<T, N>::extract_aligned() const {
  V result;

  std::memcpy(
      &result,
      reinterpret_cast<const char*>(this) +
          extract_index * sizeof(V),
      sizeof(V));

  return result;
}


template<class T, usize N>
constexpr std::tuple<
    typename vector<T, N>::half_vector,
    typename vector<T, N>::half_vector>
vector<T, N>::split() const {

  return {
      extract_aligned<half_vector, 0>(),
      extract_aligned<half_vector, 1>()
  };
}


template<class T, usize N>
constexpr typename vector<T, N>::dup_vector
vector<T, N>::dup() const {
  return dup_vector::splat(*this);
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::swizzle(
    const vector<T, N>& src) const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] = src.raw[raw[i] % N];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::swizzle(
    const vector<T, N>& src0,
    const vector<T, N>& src1) const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        (raw[i] < 2 * N)
            ? ((raw[i] < N)
                   ? src0
                   : src1)
                  .raw[raw[i] % N]
            : 0;
  }

  return result;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::swizzle(
    const vector<T, N>::mask_type& src) const {

  return std::bit_cast<vector<T, N>::mask_type>(
      swizzle(src.raw.template convert<T>()));
}


template<class T, usize N>
template<usize M>
  requires(M != N)
constexpr vector<T, N>
vector<T, N>::swizzle(
    const vector<T, M>& src) const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        raw[i] < M
            ? src.raw[raw[i]]
            : 0;
  }

  return result;
}


template<class T, usize N>
template<usize shift_amount>
constexpr vector<T, N>
vector<T, N>::shl() const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        raw[i] << shift_amount;
  }

  return result;
}


template<class T, usize N>
template<usize shift_amount>
constexpr vector<T, N>
vector<T, N>::shr() const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        raw[i] >> shift_amount;
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::andnot(
    const vector<T, N>& second) const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        raw[i] & ~second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::clamp(
    const vector<T, N>& min,
    const vector<T, N>& max) const {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        std::clamp<T>(
            raw[i],
            min.raw[i],
            max.raw[i]);
  }

  return result;
}


template<class T, usize N>
template<class U>
constexpr auto vector<T, N>::pair_dot(
    const vector<U, N>& second) const {

  using S =
      detail::signed_double_element_size_t<T>;

  vector<S, N / 2> result;

#if defined(__aarch64__)

  if constexpr (
      std::is_same_v<T, i16> &&
      std::is_same_v<U, i16> &&
      N == 8) {

    const int16x8_t a =
        vld1q_s16(raw.data());

    const int16x8_t b =
        vld1q_s16(second.raw.data());

    const int32x4_t lo =
        vmull_s16(
            vget_low_s16(a),
            vget_low_s16(b));

    const int32x4_t hi =
        vmull_s16(
            vget_high_s16(a),
            vget_high_s16(b));

    const int32x4_t sum =
        vpaddq_s32(lo, hi);

    vst1q_s32(
        result.raw.data(),
        sum);

    return result;
  }

#endif

  for (usize i = 0; i < N; i += 2) {

    result.raw[i / 2] =
        static_cast<S>(raw[i]) *
            static_cast<S>(second.raw[i]) +
        static_cast<S>(raw[i + 1]) *
            static_cast<S>(second.raw[i + 1]);
  }

  return result;
}


template<class T, usize N>
template<class V1, class V2>
constexpr vector<T, N>
vector<T, N>::accumulate_pair_dot(
    const V1& first,
    const V2& second) const {

  return *this + first.pair_dot(second);
}


template<class T, usize N>
constexpr T vector<T, N>::reduce_add() const {

  T result = 0;

  for (usize i = 0; i < N; i++) {
    result += raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr T vector<T, N>::reduce_or() const {

  T result = 0;

  for (usize i = 0; i < N; i++) {
    result |= raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr T vector<T, N>::reduce_xor() const {

  T result = 0;

  for (usize i = 0; i < N; i++) {
    result ^= raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::zip_low(
    const vector<T, N>& second) const {

  static_assert(N % 2 == 0);

  vector<T, N> result;

  for (usize i = 0; i < N; i += 2) {

    result.raw[i + 0] =
        raw[i / 2];

    result.raw[i + 1] =
        second.raw[i / 2];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::zip_high(
    const vector<T, N>& second) const {

  static_assert(N % 2 == 0);

  vector<T, N> result;

  for (usize i = 0; i < N; i += 2) {

    result.raw[i + 0] =
        raw[(N + i) / 2];

    result.raw[i + 1] =
        second.raw[(N + i) / 2];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::zip_low_128lanes(
    const vector<T, N>& second) const {

  static_assert(N % 2 == 0);

  vector<T, N> result;

  constexpr usize lane_width =
      16 / sizeof(T);

  constexpr usize lane_count =
      N / lane_width;

  static_assert(
      lane_count * lane_width == N);

  for (usize lane = 0;
       lane < lane_count;
       lane++) {

    usize lane_start =
        lane * lane_width;

    for (usize i = 0;
         i < lane_width;
         i += 2) {

      result.raw[lane_start + i + 0] =
          raw[lane_start + i / 2];

      result.raw[lane_start + i + 1] =
          second.raw[lane_start + i / 2];
    }
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
vector<T, N>::zip_high_128lanes(
    const vector<T, N>& second) const {

  static_assert(N % 2 == 0);

  vector<T, N> result;

  constexpr usize lane_width =
      16 / sizeof(T);

  constexpr usize lane_count =
      N / lane_width;

  static_assert(
      lane_count * lane_width == N);

  for (usize lane = 0;
       lane < lane_count;
       lane++) {

    usize lane_start =
        lane * lane_width;

    for (usize i = 0;
         i < lane_width;
         i += 2) {

      result.raw[lane_start + i + 0] =
          raw[lane_start +
              (lane_width + i) / 2];

      result.raw[lane_start + i + 1] =
          second.raw[lane_start +
                     (lane_width + i) / 2];
    }
  }

  return result;
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::test_vm(
    const vector& second) const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] & second.raw[i]);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::bmask_type
vector<T, N>::test_bm(
    const vector& second) const {

  return test_vm(second).to_bits();
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::test(
    const vector& second) const {

  return test_vm(second);
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::eq_vm(
    const vector& second) const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] == second.raw[i]);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::eq(
    const vector& second) const {

  return eq_vm(second);
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::neq_vm(
    const vector<T, N>& second) const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] != second.raw[i]);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::neq(
    const vector<T, N>& second) const {

  return neq_vm(second);
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::gt_vm(
    const vector<T, N>& second) const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] > second.raw[i]);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::gt(
    const vector<T, N>& other) const {

  return gt_vm(other);
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::nonzeros_vm() const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] != 0);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::nonzeros() const {

  return nonzeros_vm();
}


template<class T, usize N>
constexpr usize vector<T, N>::nonzeros_count() const {

  usize result = 0;

  for (usize i = 0; i < N; i++) {
    result += raw[i] != 0;
  }

  return result;
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::zeros_vm() const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        raw[i] == 0);
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::zeros() const {

  return zeros_vm();
}


template<class T, usize N>
constexpr usize vector<T, N>::zeros_count() const {

  usize result = 0;

  for (usize i = 0; i < N; i++) {
    result += raw[i] == 0;
  }

  return result;
}


template<class T, usize N>
constexpr typename vector<T, N>::vmask_type
vector<T, N>::msb_vm() const {

  vector<T, N>::mask_type m =
      vector<T, N>::mask_type::zero();

  for (usize i = 0; i < N; i++) {
    m.set(
        i,
        detail::msb(raw[i]));
  }

  return m;
}


template<class T, usize N>
constexpr typename vector<T, N>::mask_type
vector<T, N>::msb() const {

  return msb_vm();
}


template<class T, usize N>
std::array<T, N>
vector<T, N>::to_array() const {

  return raw;
}


template<class T, usize N>
constexpr bool operator==(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  return first.raw == second.raw;
}


template<class T, usize N>
constexpr vector<T, N>
operator~(const vector<T, N>& first) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] = ~first.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>
operator&(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] & second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator&=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] &= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator|(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] | second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator|=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] |= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator^(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] ^ second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator^=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] ^= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator+(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] + second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator+=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] += second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator-(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] - second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator-=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] -= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator*(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] * second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator*=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] *= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator<<(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] << second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator<<=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] <<= second.raw[i];
  }

  return first;
}


template<class T, usize N>
constexpr vector<T, N>
operator>>(
    const vector<T, N>& first,
    const vector<T, N>& second) {

  vector<T, N> result;

  for (usize i = 0; i < N; i++) {
    result.raw[i] =
        first.raw[i] >> second.raw[i];
  }

  return result;
}


template<class T, usize N>
constexpr vector<T, N>&
operator>>=(
    vector<T, N>& first,
    const vector<T, N>& second) {

  for (usize i = 0; i < N; i++) {
    first.raw[i] >>= second.raw[i];
  }

  return first;
}


} // namespace lps::neon
