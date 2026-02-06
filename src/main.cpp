#include <CSerialPort/SerialPort.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <iostream>

#include "video/playvideo.h"

int main() {
    std::cout << "Hello World!" << std::endl;
    videowidget::playVideoThenExit("./samples/basketball.mov");
    return 0;
}
