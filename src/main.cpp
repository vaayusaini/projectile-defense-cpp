#include "CSerialPort/SerialPort.h"
#include "CSerialPort/SerialPortInfo.h"
#include "detector/Detector.h"
#include "videostream/CameraSource.h"
#include "videostream/ImageViewer.h"
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

int detect() {
  pd::CameraSource cam0(0);
  pd::CameraSource cam1(1);
  pd::ImageViewer viewer;

  pd::DetectorConfig detectorCfg;
  detectorCfg.imscale = 0.5;
  detectorCfg.bgHistory = 120;
  detectorCfg.varThreshold = 16.0;
  detectorCfg.detectShadows = false;
  detectorCfg.bgRatio = 0.98;
  detectorCfg.closeKernelSize = 3;
  detectorCfg.closeIterations = 1;
  detectorCfg.connectivity = 8;
  detectorCfg.minArea = 400;
  detectorCfg.minAspect = 0.35f;
  detectorCfg.maxAspect = 2.8f;
  detectorCfg.maxProjectiles = 6;

  pd::Detector detector0(detectorCfg);
  pd::Detector detector1(detectorCfg);

  pd::ImageFrame f0;
  pd::ImageFrame f1;
  uint64_t lastSeq0 = 0;
  uint64_t lastSeq1 = 0;

  while (true) {

    const bool ok0 = cam0.read(f0);
    const bool ok1 = cam1.read(f1);
    bool updated0 = false;
    bool updated1 = false;

    if (ok0 && f0.frame != lastSeq0) {
      lastSeq0 = f0.frame;
      detector0.process(f0);
      pd::drawProjectiles(f0, detector0.lastProjectiles());
      viewer.show("Camera 0", f0);
      updated0 = true;
    }
    if (ok1 && f1.frame != lastSeq1) {
      lastSeq1 = f1.frame;
      detector1.process(f1);
      pd::drawProjectiles(f1, detector1.lastProjectiles());
      viewer.show("Camera 1", f1);
      updated1 = true;
    }

    const int key = viewer.waitKey(1);
    if (key == 'q')
      break;

    if (!updated0 && !updated1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  return 0;
}

void writeAngleToMotor(itas109::CSerialPort &motor, int angle) {
  const std::string packet = std::to_string(angle) + "\n";

  const int size = static_cast<int>(packet.size());
  motor.writeData(packet.data(), size);
}

void sleepFor(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

void fireTriggerMotor(itas109::CSerialPort &motor) {
  writeAngleToMotor(motor, 65);
  sleepFor(1000);

  writeAngleToMotor(motor, 35);
}

int arduino() {
  using itas109::CSerialPort;

  const std::vector<itas109::SerialPortInfo> ports = itas109::CSerialPortInfo::availablePortInfos();

  if (ports.empty()) {
    std::cerr << "No serial ports found.\n";
    return -1;
  }

  for (int i = 0; i < ports.size(); i++) {
    std::cout << "port: " << ports[i].portName << " id: " << ports[i].hardwareId << std::endl;
  }

  std::string portName;

  std::cout << "enter the motor port name:";
  std::cin >> portName;

  itas109::CSerialPort motor(portName.c_str());
  motor.setBaudRate(9600);
  motor.open();

  if (!motor.isOpen()) {
    std::cout << "unable to connect to motor " << portName;
    return -1;
  }

  sleepFor(100);

  while (true) {
    std::cout << "enter input:";

    std::string input;
    std::cin >> input;

    if (input == "q") {
      break;
    }

    fireTriggerMotor(motor);
  }

  // while (true) {
  //   fireTriggerMotor(motor);
  //   sleepFor(1000);

  // for (int angle = 35; angle <= 85; angle++) {
  //   writeAngleToMotor(motor, angle);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // }

  // for (int angle = 85; angle >= 35; angle--) {
  //   writeAngleToMotor(motor, angle);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // }
  // }

  return 0;
}

int main() { return arduino(); }
