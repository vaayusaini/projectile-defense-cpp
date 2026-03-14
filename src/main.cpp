#include "CSerialPort/SerialPort.h"
#include "CSerialPort/SerialPortInfo.h"
#include "detector/Detector.h"
#include "tracker/Tracker.h"
#include "videostream/CameraSource.h"
#include "videostream/ImageViewer.h"
#include <chrono>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

itas109::CSerialPort arduinoPort;


int detect() {
  pd::CameraSource cam0(1);
  pd::CameraSource cam1(2);
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
  detectorCfg.minArea = 260;
  detectorCfg.minAspect = 0.35f;
  detectorCfg.maxAspect = 2.8f;
  detectorCfg.maxProjectiles = 6;

  pd::Detector detector0(detectorCfg);
  pd::Detector detector1(detectorCfg);

  pd::ImageFrame f0;
  pd::ImageFrame f1;
  uint64_t lastSeq0 = 0;
  uint64_t lastSeq1 = 0;

  std::vector<pd::ProjectileFrame> cam0Projectiles;
  std::vector<pd::ProjectileFrame> cam1Projectiles;

  pd::Tracker tracker;

  while (true) {

    const bool ok0 = cam0.read(f0);
    const bool ok1 = cam1.read(f1);

    if (!ok0 || !ok1) {
      const int key = viewer.waitKey(1);
      if (key == 'q')
        break;

      continue;
    }

    if (f0.frame == lastSeq0 || f1.frame == lastSeq1) {
      const int key = viewer.waitKey(1);
      if (key == 'q')
        break;

      continue;
    }

    lastSeq0 = f0.frame;
    detector0.process(f0);
    cam0Projectiles = detector0.lastProjectiles();

    for (int i = 0; i < cam0Projectiles.size(); i++) {
      pd::ProjectileFrame &projectile = cam0Projectiles[i];
      projectile.center.x = (1920 - projectile.center.x);
    }

    pd::drawProjectiles(f0, detector0.lastProjectiles());
    viewer.show("Camera 0", f0);

    lastSeq1 = f1.frame;
    detector1.process(f1);
    cam1Projectiles = detector1.lastProjectiles();

    for (int i = 0; i < cam1Projectiles.size(); i++) {
      pd::ProjectileFrame &projectile = cam1Projectiles[i];
      projectile.center.x = (1920 - projectile.center.x);
    }

    pd::drawProjectiles(f1, detector1.lastProjectiles());
    viewer.show("Camera 1", f1);

    std::cout << "frame0: " << lastSeq0 << " frame1: " << lastSeq1 << std::endl;
    tracker.updateCoordinates(cam0Projectiles, cam1Projectiles);

    double t = lastSeq1 / 30.0;
    tracker.print(t);
    //NEED TO ADD PORT NAMES FOR THE TWO MOTORS BELOW
    writeAngleToMotor(arduinoPort, static_cast<int>(tracker.releaseAngle * 180 / 3.14159));
    if ((t - tracker.releaseTime) < 0.04) {
      sleepFor((t - tracker.releaseTime) * 1000);
      fireTriggerMotor(arduinoPort);
    }
  }

  return 0;
}
// VAAYU YOU NEED TO FIX THIS I DONT THINK ITLL WORK
void writeAngleToMotor(itas109::CSerialPort &motor, int angle) {
  const std::string packet = std::to_string(angle) + "\n";

  const int size = static_cast<int>(packet.size());
  motor.writeData(packet.data(), size);
}

void sleepFor(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

void fireTriggerMotor(itas109::CSerialPort &motor) {
  writeAngleToMotor(motor, 120);
  sleepFor(1000);

  writeAngleToMotor(motor, 30);
}

long long getCurrentTimeMilliseconds() {
  // Get the current time point from the system clock
  auto now = std::chrono::system_clock::now();

  // Cast the duration since the epoch to milliseconds
  auto milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());

  // Return the count of milliseconds as a long long integer
  return milliseconds_since_epoch.count();
}

int trigger() {
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

    writeAngleToMotor(motor, 120);

    long long timeStarted = getCurrentTimeMilliseconds();
    long long timeElapsed = 0;

    while (timeElapsed < 1000) {
      timeElapsed = getCurrentTimeMilliseconds() - timeStarted;
      std::cout << "time elapsed: " << timeElapsed << std::endl;
      sleepFor(1);
    }

    writeAngleToMotor(motor, 30);
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

int platformMotor() {
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

    int inputAngle = std::stoi(input);
    writeAngleToMotor(motor, inputAngle);
  }

  return 0;
}

int main() { return detect(); }
