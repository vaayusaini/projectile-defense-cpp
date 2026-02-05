cmake_minimum_required(VERSION 4.1)
project(projectile-defense LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.h)
add_executable(projectile-defense MACOSX_BUNDLE ${SRC_FILES})

# vcpkg toolchain makes find_package() work out of the box
find_package(OpenCV CONFIG REQUIRED)
find_package(cserialport CONFIG REQUIRED)

# Link OpenCV components
target_include_directories(projectile-defense PRIVATE ${CSerialPort_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/external/metal-cpp)
target_link_libraries(projectile-defense PRIVATE ${OpenCV_LIBS} ${CSerialPort_Library} "-framework Metal" "-framework Foundation" "-framework QuartzCore")

# Build as a macOS .app bundle
set_target_properties(projectile-defense PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.vaayu.projectile-defense"
    MACOSX_BUNDLE_BUNDLE_NAME "Projectile Defense"
)
