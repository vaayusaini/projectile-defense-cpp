cmake_minimum_required(VERSION 4.1)
project(projectile-defense LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(projectile-defense MACOSX_BUNDLE
    src/main.cpp
    src/projectiledetector.cpp
    src/projectiledetector.h
    src/position.cpp
    src/position.h)

# vcpkg toolchain makes find_package() work out of the box
find_package(OpenCV REQUIRED)

# Link OpenCV components
target_link_libraries(projectile-defense PRIVATE ${OpenCV_LIBS})

# Build as a macOS .app bundle
set_target_properties(projectile-defense PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.vaayu.projectile-defense"
    MACOSX_BUNDLE_BUNDLE_NAME "Projectile Defense"
)