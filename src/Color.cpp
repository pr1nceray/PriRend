#include "Color.h"

/*
*
* This file contains operators for Color class
*
*/


/*
* Defines operators +, -, +=, -= for Color
*/

Color Color::operator+(const Color & rhs) {
  return Color(r + rhs.r, g + rhs.g, b + rhs.b);
}

Color Color::operator-(const Color & rhs) {
  return Color(r - rhs.r, g - rhs.g, b - rhs.b);
}

void Color::operator+=(const Color & rhs) {
    r += rhs.r;
    g += rhs.g;
    b += rhs.b;
}

void Color::operator-=(const Color & rhs) {
    r -= rhs.r;
    g -= rhs.g;
    b -= rhs.b;
}


/*
* Defines operators +, -, *, /, +=, -=, *=, /= for a float.
*/

Color Color::operator+(const float & rhs) {
    return Color(r + rhs, g + rhs, b + rhs);
}

Color Color::operator-(const float & rhs) {
    return Color(r - rhs, g -rhs, b - rhs);
}

Color Color::operator*(const float & rhs) {
    return Color(r * rhs, g * rhs, b * rhs);
}

Color Color::operator/(const float & rhs) {
    return Color(r/rhs, g/rhs, b/rhs);
}

void Color::operator+=(const float & rhs) {
    r += rhs;
    g += rhs;
    b += rhs;
}

void Color::operator-=(const float & rhs) {
    r -= rhs;
    g -= rhs;
    b -= rhs;
}

void Color::operator*=(const float & rhs) {
    r *= rhs;
    g *= rhs;
    b *= rhs;
}

void Color::operator/=(const float & rhs) {
    r /= rhs;
    g /= rhs;
    b /= rhs;
}
