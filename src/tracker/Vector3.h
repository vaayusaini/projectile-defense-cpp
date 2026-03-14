#pragma once
#include <ostream>
#include <cmath>

namespace pd {

struct Vector3 {
  double x = 0;
  double y = 0;
  double z = 0;

  Vector3() = default;
  Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

  Vector3 operator+(const Vector3 &other) const { return Vector3(x + other.x, y + other.y, z + other.z); }
  Vector3 operator-(const Vector3 &other) const { return Vector3(x - other.x, y - other.y, z - other.z); }
  Vector3 &operator+=(const Vector3 &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
  Vector3 &operator-=(const Vector3 &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }
  Vector3 operator*(auto value) const { return Vector3(x * value, y * value, z * value); }
  Vector3 operator/(auto value) const { return Vector3(x / value, y / value, z / value); }
  double magnitude() const { return std::sqrt(x*x + y*y + z*z); }
  double magnitudeSquared() const { return (x*x + y*y + z*z); }
  friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

} // namespace pd