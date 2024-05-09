#pragma once
#include <cstdint>

struct Color{
  uint8_t r;
  uint8_t g;
  uint8_t b;

  Color() :
  r(0), g(0), b(0) {
  }

  Color(uint16_t r_in, uint16_t g_in, uint16_t b_in) :
  r(r_in), g(g_in), b(b_in) {
  }

  /*
  * Operators defined in Color.cpp
  */
  Color operator+(const Color & rhs);
  Color operator-(const Color & rhs);

  void operator+=(const Color & rhs);
  void operator-=(const Color & rhs);

  void operator+=(const float & rhs);
  void operator-=(const float & rhs);
  void operator*=(const float & rhs);
  void operator/=(const float & rhs);

  Color operator+(const float & rhs);
  Color operator-(const float & rhs);
  Color operator*(const float & rhs);
  Color operator/(const float & rhs);
};
