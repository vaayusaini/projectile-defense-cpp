#pragma once

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
  Vector3 operator/(double value) const { return Vector3(x / value, y / value, z / value); }
  bool operator<(const Vector3 &other) const {
    return ((x * x + y * y + z * z) < (other.x * other.x + other.y * other.y + other.z * other.z));
  }
};

} // namespace pd