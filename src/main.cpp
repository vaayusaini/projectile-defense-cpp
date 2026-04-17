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

// VAAYU YOU NEED TO FIX THIS I DONT THINK ITLL WORK
void writeAngleToMotor(itas109::CSerialPort &motor, int angle) {
  const std::string packet = std::to_string(angle) + "\n";

  const int size = static_cast<int>(packet.size());
  motor.writeData(packet.data(), size);
}

void trigger(itas109::CSerialPort &motor) {
  const std::string packet = "f\n";

  const int size = static_cast<int>(packet.size());
  motor.writeData(packet.data(), size);
}

void sleepFor(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

itas109::CSerialPort getMotor() {
  using itas109::CSerialPort;

  const std::vector<itas109::SerialPortInfo> ports = itas109::CSerialPortInfo::availablePortInfos();

  if (ports.empty()) {
    std::cerr << "No serial ports found.\n";
    std::runtime_error("Could not find any serial ports");
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

  return motor;
}

int detect() {

  itas109::CSerialPort motor = getMotor();

  if (!motor.isOpen()) {
    std::cout << "unable to connect to motor ";
    return -1;
  }

  sleepFor(100);

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
  detectorCfg.minArea = 200;
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

  std::vector<double> frameTimes;

  bool hasFired = false;

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

    // for (int i = 0; i < cam0Projectiles.size(); i++) {
    //   pd::ProjectileFrame &projectile = cam0Projectiles[i];
    //   projectile.center.x = (1920 - projectile.center.x);
    // }

    if (cam0Projectiles.size() == 2) {
      pd::ProjectileFrame largestProjectile;
      int largestArea = 0;
      for (int i = 0; i < cam0Projectiles.size(); i++) {
        pd::ProjectileFrame &projectile = cam0Projectiles[i];
        if (projectile.bbox.area > largestArea) {
          largestArea = projectile.bbox.area;
          largestProjectile = projectile;
        }
      }

      cam0Projectiles = {largestProjectile};
    }

    pd::drawProjectiles(f0, detector0.lastProjectiles());
    viewer.show("Camera 0", f0);

    lastSeq1 = f1.frame;
    detector1.process(f1);
    cam1Projectiles = detector1.lastProjectiles();

    // for (int i = 0; i < cam1Projectiles.size(); i++) {
    //   pd::ProjectileFrame &projectile = cam1Projectiles[i];
    //   projectile.center.x = (1920 - projectile.center.x);
    // }

    if (cam1Projectiles.size() == 2) {
      pd::ProjectileFrame largestProjectile;
      int largestArea = 0;
      for (int i = 0; i < cam1Projectiles.size(); i++) {
        pd::ProjectileFrame &projectile = cam1Projectiles[i];
        if (projectile.bbox.area > largestArea) {
          largestArea = projectile.bbox.area;
          largestProjectile = projectile;
        }
      }

      cam1Projectiles = {largestProjectile};
    }

    pd::drawProjectiles(f1, detector1.lastProjectiles());
    viewer.show("Camera 1", f1);

    // std::cout << "frame0: " << lastSeq0 << " frame1: " << lastSeq1 << std::endl;
    tracker.updateCoordinates(cam0Projectiles, cam1Projectiles);

    double t = lastSeq1 / 60.0;
    // frameTimes.push_back(t);
    // while (t - frameTimes[0] > 1) {
    //   frameTimes.
    // }

    // tracker.print(t);
    // NEED TO ADD PORT NAMES FOR THE TWO MOTORS BELOW
    writeAngleToMotor(motor, static_cast<int>(-tracker.releaseAngle * 180 / 3.14159 + 90));

    std::cout << "t: " << t << " releaseTime: " << tracker.releaseTime << std::endl;

    if (!hasFired && tracker.releaseTime > 0) {
      if (tracker.releaseTime < t) {
        std::cout << "has fired" << std::endl;
        trigger(motor);
        hasFired = true;
      } else {
        if (((t - tracker.releaseTime) < 0.04)) {
          sleepFor((t - tracker.releaseTime) * 1000);

          std::cout << "has fired" << std::endl;
          trigger(motor);
          hasFired = true;
        }
      }
    }

    // if (((t - tracker.releaseTime) < 0.04) and (tracker.releaseTime != 0)) {
    //   sleepFor((t - tracker.releaseTime) * 1000);
    //   std::cout << "wanted to fire" << std::endl;
    //   // trigger(motor);
    // }
  }

  return 0;
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

itas109::CSerialPort askForMotor() {
  const std::vector<itas109::SerialPortInfo> ports = itas109::CSerialPortInfo::availablePortInfos();

  if (ports.empty()) {
    std::cerr << "No serial ports found.\n";
    std::runtime_error("Could not find any serial ports");
  }

  for (int i = 0; i < ports.size(); i++) {
    std::cout << "port: " << ports[i].portName << " id: " << ports[i].hardwareId << std::endl;
  }

  std::string portIndexString;

  std::cout << "enter the port index:";
  std::cin >> portIndexString;

  int portIndex = std::stoi(portIndexString);

  std::string portName = ports[portIndex].portName;
  itas109::CSerialPort motor(portName.c_str());
  motor.setBaudRate(9600);
  motor.open();

  return motor;
}

int manualControl() {
  itas109::CSerialPort motor = getMotor();

  if (!motor.isOpen()) {
    std::cout << "unable to connect to motor";
    return -1;
  }

  sleepFor(100);

  while (true) {
    std::cout << "enter input:";

    std::string input;
    std::cin >> input;

    if (input == "q") {
      break;
    } else if (input == "f") {
      trigger(motor);
    } else {
      int inputAngle = std::stoi(input);
      writeAngleToMotor(motor, inputAngle);
    }
  }

  return 0;
}

int main() { return detect(); }
