#include "Color.h"

/*
*
* This file contains operators for Color class
*
*/


/*
* Defines operators +, -, +=, -= for Color
*/

__host__ __device__ Color Color::operator+(const Color & rhs) {
  return Color(r + rhs.r, g + rhs.g, b + rhs.b);
}

__host__ __device__ Color Color::operator-(const Color & rhs) {
  return Color(r - rhs.r, g - rhs.g, b - rhs.b);
}

__host__ __device__ Color Color::operator*(const Color & rhs) {
    return Color(r * rhs.r, g * rhs.g, b * rhs.b);
}


__host__ __device__ void Color::operator+=(const Color & rhs) {
    r += rhs.r;
    g += rhs.g;
    b += rhs.b;
}

__host__ __device__ void Color::operator-=(const Color & rhs) {
    r -= rhs.r;
    g -= rhs.g;
    b -= rhs.b;
}


/*
* Defines operators +, -, *, /, +=, -=, *=, /= for a float.
*/

__host__ __device__ Color Color::operator+(const float & rhs) {
    return Color(r + rhs, g + rhs, b + rhs);
}

__host__ __device__ Color Color::operator-(const float & rhs) {
    return Color(r - rhs, g -rhs, b - rhs);
}

__host__ __device__ Color Color::operator*(const float & rhs) {
    return Color(r * rhs, g * rhs, b * rhs);
}

__host__ __device__ Color Color::operator/(const float & rhs) {
    return Color(r/rhs, g/rhs, b/rhs);
}

__host__ __device__ void Color::operator+=(const float & rhs) {
    r += rhs;
    g += rhs;
    b += rhs;
}

__host__ __device__ void Color::operator-=(const float & rhs) {
    r -= rhs;
    g -= rhs;
    b -= rhs;
}

__host__ __device__ void Color::operator*=(const float & rhs) {
    r *= rhs;
    g *= rhs;
    b *= rhs;
}

__host__ __device__ void Color::operator/=(const float & rhs) {
    r /= rhs;
    g /= rhs;
    b /= rhs;
}
