#include "rose/util/string.hpp"

#include "rose/common.hpp"

#if defined(__ANDROID__)
#include <cerrno>
#include <cstdlib>
#else
#include <charconv>
#endif
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rose {

  auto string_split(std::string text, char delim) -> std::vector<std::string> {
    std::string line;
    std::vector<std::string> v;
    std::stringstream ss(text);
    while (std::getline(ss, line, delim))
      v.push_back(line);
    return v;
  }

  template<typename Int>
  auto parse_integer(std::string_view str) -> std::optional<Int> {
    bool negate = false;
    Int result = 0;
    usize i = 0;

    if (str.size() == 0)
      return std::nullopt;

    if (str[0] == '-') {
      if (!std::is_signed_v<Int> || str.size() == 1)
        return std::nullopt;
      negate = true;
      i = 1;
    }

    for (; i < str.size(); i++) {
      if (str[i] < '0' && str[i] > '9')
        return std::nullopt;
      if (__builtin_mul_overflow(result, static_cast<Int>(10), &result))
        return std::nullopt;
      if (__builtin_add_overflow(result, static_cast<Int>(str[i] - '0'), &result))
        return std::nullopt;
    }

    if constexpr (std::is_signed_v<Int>) {
      return negate ? -result : result;
    } else {
      return result;
    }
  }

  auto parse_int(std::string_view str) -> std::optional<int> {
    return parse_integer<int>(str);
  }

  auto parse_u16(std::string_view str) -> std::optional<u16> {
    return parse_integer<u16>(str);
  }

  auto parse_u32(std::string_view str) -> std::optional<u32> {
    return parse_integer<u32>(str);
  }

  auto parse_u64(std::string_view str) -> std::optional<u64> {
    return parse_integer<u64>(str);
  }

  auto parse_i32(std::string_view str) -> std::optional<i32> {
    return parse_integer<i32>(str);
  }

  auto parse_usize(std::string_view str) -> std::optional<usize> {
    return parse_integer<usize>(str);
  }

  auto parse_f64(std::string_view str) -> std::optional<f64> {
#if defined(__ANDROID__)
    if (str.empty())
      return std::nullopt;

    // NDK r27c libc++ lacks the floating-point std::from_chars overload.
    // Require full input consumption and reject range errors like the
    // upstream from_chars implementation.
    const std::string text {str};
    errno = 0;
    char* end = nullptr;
    const f64 result = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || errno == ERANGE)
      return std::nullopt;
    return result;
#else
    f64 result;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
    if (ec != std::errc {} || ptr != str.data() + str.size())
      return std::nullopt;
    return result;
#endif
  }

}  // namespace rose
