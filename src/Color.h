#include <cstdint>

struct Color
{
  uint8_t r;
  uint8_t g;
  uint8_t b;     
  
  Color() :
  r(0), g(0), b(0)
  { 
  } 

  Color(uint8_t r_in, uint8_t g_in, uint8_t b_in) :
  r(r_in), g(g_in), b(b_in)
  { 
  } 

};

Color operator+(const Color & lhs, const Color & rhs)
{
    return Color(lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b);
}

Color & operator+=(Color & lhs, const Color & rhs)
{
    lhs.r += rhs.r;
    lhs.g += rhs.g;
    lhs.b += rhs.b;
    return lhs;
}