#include "Color.cuh"

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

/*
* Clamp the color to be a valid color.
*/
__device__ Color clampColor(Color final) {
    uint8_t finalR = final.r > 255? 255 : static_cast<uint8_t>(final.r);
    uint8_t finalG = final.g > 255? 255 : static_cast<uint8_t>(final.g);
    uint8_t finalB = final.b > 255? 255 : static_cast<uint8_t>(final.b);
    return Color(finalR, finalG, finalB);
}

