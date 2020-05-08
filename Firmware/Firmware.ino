/*************
   Somatic data glove firmware
   by Zack Freedman of Voidstar Lab
   (c) Voidstar Lab 2019
   Somatic is licensed Creative Commons 4.0 Attribution Non-Commercial

   Uses code by Paul Joffressen, Kris Winer, and the Arduino project

   Deploy to Teensy 4.0 at max clock speed
*/

// The following dumb bullshit just lets us turn off serial debug
#define DEBUG
#define GET_MACRO(_1, _2, NAME, ...) NAME

#ifdef DEBUG
#define debug_print_formatted(x, y) Serial.print(x,y)
#define debug_print_noformat(x) Serial.print(x)
#define debug_println_formatted(x, y) Serial.println(x, y)
#define debug_println_noformat(x) Serial.println(x)
#define debug_write(x) Serial.write(x)
#else
#define debug_print_formatted(x, y)
#define debug_print_noformat(x)
#define debug_println_formatted(x, y)
#define debug_println_noformat(x)
#define debug_write(x)
#endif

#define debug_print(...) GET_MACRO(__VA_ARGS__, debug_print_formatted, debug_print_noformat)(__VA_ARGS__)
#define debug_println(...) GET_MACRO(__VA_ARGS__, debug_println_formatted, debug_println_noformat)(__VA_ARGS__)
#define sanity(x) debug_println("Sanity " #x)

// #define debug_angular_velocity
#define debug_dump_bt_rx

#define bt Serial1

// End dumb debugging bullshit

#include <Wire.h>
#include "imu.h"

unsigned long fingerDebounce = 100;

const byte btRtsPin = 2;
const byte vibePin = 23;
const byte fingerSwitchPins[] = {9, 10, 11, 12};
bool lastFingerPositions[4];
unsigned long lastFingerStableTimestamps[4];

float lastBearing[3];

const float velocityThresholdToBegin = 4. / 1000000.;  // In rad/microsecond
const float velocityThresholdToEnd = 2. / 1000000.;  // Also in rad/microsecond
const int historyLength = 10;
float angularVelocityWindow[historyLength];

float bearingAtLastBuzz[2];
const float distanceBetweenBuzzes = 15. / 180. * PI;
elapsedMillis timeSinceBuzzStarted;
unsigned long currentBuzzDuration;

bool isGesturing;

elapsedMillis timeSinceLastGesture;
unsigned long lastTimestamp;

IMU imu = IMU();

elapsedMillis timeSinceLastDebugCommandChar;
#define commandLockout 10000

bool isFrozen = false;
elapsedMillis timeSinceFreeze;
#define freezeTime 2000

char outgoingPacket[100];

void setup() {
  Serial.begin(115200);
  bt.begin(57600);

  analogWriteFrequency(vibePin, 93750);  // Set to high frequency so switching noise is inaudible

  pinMode(btRtsPin, INPUT);

  // Some butt-head didn't leave room for i2c pullups on the lil board.
  // Using adjacent pins as makeshift 3V3 sources :|
  pinMode(17, OUTPUT);
  digitalWrite(17, HIGH);
  pinMode(20, OUTPUT);
  digitalWrite(20, HIGH);

  pinMode(vibePin, OUTPUT);

  Wire.begin();
  Wire.setClock(400000);
  imu.setup();

  for (int i = 0; i < 4; i++) {
    pinMode(fingerSwitchPins[i], INPUT_PULLUP);
  }

  lastTimestamp = micros();
}

void loop() {
  unsigned long timestamp = micros();
  float sampleRate = timestamp - lastTimestamp;

  if (timeSinceBuzzStarted > currentBuzzDuration) {
    analogWrite(vibePin, 0);
  }

  while (Serial.available()) {
    int incoming = Serial.read();
    //    Serial.write(incoming);
    bt.write(incoming);

    timeSinceLastDebugCommandChar = 0;
  }

  while (bt.available()) {
    if (bt.peek() == '>') {
      if (bt.available() >= 5
          && bt.read() == '>'
          && bt.read() == 'A'
          && bt.read() == 'T'
          && bt.read() == '\r'
          && bt.read() == '\n') {
        debug_println("Got AT - request to acknowledge");
        bt.print(">OK\n");
      }
      else break;
    }
    else {
#ifdef debug_dump_bt_rx
      if (timeSinceLastDebugCommandChar < commandLockout)
        timeSinceLastDebugCommandChar = 0;
      Serial.write(bt.read());
#endif
    }
  }

  // TODO: Handle %CONNECT,5800E382649E,0 and %DISCONNECT messages from bt

  if (imu.poll() & 0x04) {  // Only send data when we have new AHRS
    float theta = 0;

    theta = asin(norm(lastBearing[0], lastBearing[1], imu.Quat[0], imu.Quat[1]));
    float angularVelocity = theta / sampleRate;

#ifdef debug_angular_velocity
    debug_print("Theta: ");
    debug_println(theta);
    debug_print("Angular velocity (per sec): ");
    debug_println(angularVelocity * 1000000.);
#endif

    for (int i = historyLength - 2; i >= 0; i--) {
      angularVelocityWindow[i + 1] = angularVelocityWindow[i];
    }

    angularVelocityWindow[0] = angularVelocity;

    //   Packet format:
    //   >[./|],[./|],[./|],[./|],[float h],[float p],[float r],[float accel x],[accel y],[accel z],[us since last sample]

    for (int i = 0; i < 4; i++) {
      bool fingerPosition = digitalRead(fingerSwitchPins[i]);
      if (fingerPosition == lastFingerPositions[i]) {
        lastFingerStableTimestamps[i] = millis();
      }
      else if (isGesturing || millis() - lastFingerStableTimestamps[i] >= fingerDebounce) {
        lastFingerPositions[i] = fingerPosition;
        lastFingerStableTimestamps[i] = millis();
      }
    }

    bool handIsMoving = false;

    for (int i = 0; i < historyLength; i++) {
      if ((isGesturing && angularVelocityWindow[i] >= velocityThresholdToEnd)
          || (!isGesturing && angularVelocityWindow[i] >= velocityThresholdToBegin)) {
        handIsMoving = true;
        break;
      }
    }

    if (handIsMoving) {
      if (lastFingerPositions[0] && lastFingerPositions[1] && lastFingerPositions[2] && !lastFingerPositions[3]) {
        if (!isGesturing) {
          buzzFor(250, 20);
          bearingAtLastBuzz[0] = imu.Quat[0];
          bearingAtLastBuzz[1] = imu.Quat[1];
        }
        isGesturing = true;
      }
    }
    else {
      isGesturing = false;
    }

    if (timeSinceBuzzStarted >= 80
        && isGesturing
        && norm(bearingAtLastBuzz[0], bearingAtLastBuzz[1], imu.Quat[0], imu.Quat[1]) >= distanceBetweenBuzzes) {
      buzzFor(250, 20);
      bearingAtLastBuzz[0] = imu.Quat[0];
      bearingAtLastBuzz[1] = imu.Quat[1];
    }

    if (timeSinceLastDebugCommandChar >= commandLockout) {
      if (!digitalRead(btRtsPin)) {
        for (int i = 0; i < 100; i++) outgoingPacket[i] = 0;

        outgoingPacket[0] = '>';

        if (lastFingerPositions[0]) outgoingPacket[1] = '.';
        else outgoingPacket[1] = '|';
        outgoingPacket[2] = ',';

        if (lastFingerPositions[1]) outgoingPacket[3] = '.';
        else outgoingPacket[3] = '|';
        outgoingPacket[4] = ',';

        if (lastFingerPositions[2]) outgoingPacket[5] = '.';
        else outgoingPacket[5] = '|';
        outgoingPacket[6] = ',';

        if (lastFingerPositions[3]) outgoingPacket[7] = '.';
        else outgoingPacket[7] = '|';
        outgoingPacket[8] = ',';

        //      bt.print('>');
        //      bt.print(lastFingerPositions[0] ? '.' : '|');
        //      bt.print(',');
        //      bt.print(lastFingerPositions[1] ? '.' : '|');
        //      bt.print(',');
        //      bt.print(lastFingerPositions[2] ? '.' : '|');
        //      bt.print(',');
        //      bt.print(lastFingerPositions[3] ? '.' : '|');
        //      bt.print(',');

        dtostrf(imu.Quat[0] * -1, 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.Quat[1], 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.Quat[2], 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        //      bt.print(imu.Quat[0] * -1, 4);
        //      bt.print(',');
        //
        //      bt.print(imu.Quat[1], 4);
        //      bt.print(',');
        //
        //      bt.print(imu.Quat[2], 4);
        //      bt.print(',');

        dtostrf(imu.ax, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.ay, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.az, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        //      bt.print(imu.ax, 4);
        //      bt.print(',');
        //
        //      bt.print(imu.ay, 4);
        //      bt.print(',');
        //
        //      bt.print(imu.az, 4);
        //      bt.print(',');

        itoa(sampleRate, &outgoingPacket[strlen(outgoingPacket)], 10);
        outgoingPacket[strlen(outgoingPacket)] = '\n';

        //      bt.println(sampleRate);

        for (int i = 0; i < strlen(outgoingPacket); i++) {
          if (digitalRead(btRtsPin)) {
            debug_println("Agh! RTS! Abandon ship!");
            break;
          }
          else bt.print(outgoingPacket[i]);
        }

        if (isFrozen) {
          debug_println("Back to normal");
          isFrozen = false;
        }
      }
      else {
        if (!isFrozen) {
          debug_println("BOO! RTS! Can't send");
          timeSinceFreeze = 0;
          isFrozen = true;
        }
        else {
          if (timeSinceFreeze >= freezeTime) {
            debug_println("Kicking the module");
            bt.print('\n');
            timeSinceFreeze = 0;
          }
        }
      }
    }

    lastTimestamp = timestamp;
    lastBearing[0] = imu.Quat[0];
    lastBearing[1] = imu.Quat[1];
    lastBearing[2] = imu.Quat[2];
  }
}

void quaternionMultiply(float* q1, float* q2, float* out) {
  out[0] =  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
  out[1] = -q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
  out[2] =  q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3] + q1[3] * q2[2];
  out[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3];
}

//void configureBluetooth() {
//  while (bt.available()) bt.read();
//
//  bt.print("$$$");
//  delay(50);
//
//  while (bt.available()) {
//    if (bt.read() == 'O')
//  }
//}

float wrappedDelta(float oldValue, float newValue) {
  float delta = oldValue - newValue;

  if (delta > PI)
    delta -= 2 * PI;
  else if (delta < -PI)
    delta += 2 * PI;

  return delta;
}

void buzzFor(unsigned int strength, unsigned long duration) {
  analogWrite(vibePin, strength);
  timeSinceBuzzStarted = 0;
  currentBuzzDuration = duration;
}

float norm(float oldX, float oldY, float newX, float newY) {
  float xDelta = wrappedDelta(oldX, newX);
  float yDelta = wrappedDelta(oldY, newY);

  return sqrt(xDelta * xDelta + yDelta * yDelta);
}

