cmake_minimum_required(VERSION 4.1)
project(projectile-defense LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile all .cpp/.mm (headers don't need to be compiled, but globbing them is harmless)
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/src/*.mm"
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.hpp"
)

# Plain executable (NOT a macOS bundle)
add_executable(projectile-defense ${SRC_FILES})

# vcpkg toolchain makes find_package() work out of the box
find_package(OpenCV CONFIG REQUIRED)
find_package(cserialport CONFIG REQUIRED)

message(STATUS "CSerialPort_LIBRARY='${CSerialPort_LIBRARY}'")

target_include_directories(projectile-defense PRIVATE
  ${CSerialPort_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/external/metal-cpp
)

target_link_libraries(projectile-defense PRIVATE
  ${OpenCV_LIBS}
  ${CSerialPort_LIBRARY}
  "-framework IOKit"
  "-framework AVFoundation"
  "-framework AVKit"
  "-framework AppKit"
  "-framework CoreMedia"
  "-framework CoreVideo"
  "-framework CoreGraphics"
  "-framework QuartzCore"
  "-framework Metal"
  "-framework MetalKit"
  "-framework Foundation"
  "-framework Cocoa"
)

# Enable ARC for Objective-C++ files
target_compile_options(projectile-defense PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)