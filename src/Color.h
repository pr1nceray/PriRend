#pragma once
#include <cstdint>
#include "cuda_runtime.h"

struct Color{
  float r;
  float g;
  float b;

   __host__ __device__ __inline__ Color() :
  r(0), g(0), b(0) {
  }

  __host__ __device__ __inline__ Color(float r_in, float g_in, float b_in) :
  r(r_in), g(g_in), b(b_in) {
  }

  /*
  * Operators defined in Color.cpp
  */
  __host__ __device__ Color operator+(const Color & rhs);
  __host__ __device__ Color operator-(const Color & rhs);
  __host__ __device__ Color operator*(const Color & rhs);

  __host__ __device__ void operator+=(const Color & rhs);
  __host__ __device__ void operator-=(const Color & rhs);

  __host__ __device__ void operator+=(const float & rhs);
  __host__ __device__ void operator-=(const float & rhs);
  __host__ __device__ void operator*=(const float & rhs);
  __host__ __device__ void operator/=(const float & rhs);

  __host__ __device__ Color operator+(const float & rhs);
  __host__ __device__ Color operator-(const float & rhs);
  __host__ __device__ Color operator*(const float & rhs);
  __host__ __device__ Color operator/(const float & rhs);
};
