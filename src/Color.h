#pragma once
#include <cstdint>

struct Color{
  float r;
  float g;
  float b;

  Color() :
  r(0), g(0), b(0) {
  }

  Color(float r_in, float g_in, float b_in) :
  r(r_in), g(g_in), b(b_in) {
  }

  /*
  * Operators defined in Color.cpp
  */
  Color operator+(const Color & rhs);
  Color operator-(const Color & rhs);
  Color operator*(const Color & rhs);

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
