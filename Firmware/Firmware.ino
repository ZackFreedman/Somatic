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

#define bt Serial

// End dumb debugging bullshit

//#include <i2c_t3.h>
#include <Wire.h>
#include "imu.h"

unsigned long fingerDebounce = 100;

const byte vibePin = 23;
const byte fingerSwitchPins[] = {9, 10, 11, 12};
bool lastFingerPositions[4];
unsigned long lastFingerStableTimestamps[4];

float lastBearing[3];

const float velocityThresholdToBegin = 2. / 1000000.;  // In rad/microsecond
const float velocityThresholdToEnd = 1 / 1000000.;  // Also in rad/microsecond
const int historyLength = 5;
float angularVelocityWindow[historyLength];

float bearingAtLastBuzz[2];
const float distanceBetweenBuzzes = 15. / 180. * PI;
elapsedMillis timeSinceBuzzStarted;
unsigned long currentBuzzDuration;

bool isGesturing;

elapsedMillis timeSinceLastGesture;
unsigned long lastTimestamp;

IMU imu = IMU();

void setup() {
  //  Wire.beginTransmission(0x24);
  //  Wire.write(0x03); // Select control register
  //  Wire.write(0x3F); // Assign pins 0-5 to inputs
  //  byte error = Wire.endTransmission();

  //  Wire.begin(I2C_MASTER, 0x00, I2C_PINS_18_19, I2C_PULLUP_INT, I2C_RATE_400);
  Serial.begin(115200);
  bt.begin(115200);

  analogWriteFrequency(vibePin, 93750);

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

  if (bt.available() >= 5) {
    for (int i = 0; i < bt.available(); i++) {
      if (bt.read() == '>'
          && bt.read() == 'A'
          && bt.read() == 'T'
          && bt.read() == '\r'
          && bt.read() == '\n') {
        bt.println(">OK");
      }
    }
  }

  // TODO: Handle %CONNECT,5800E382649E,0 and %DISCONNECT messages from bt

  if (imu.poll() & 0x04) {  // Only send data when we have new AHRS
    float theta = 0;

    theta = asin(norm(lastBearing[0], lastBearing[1], imu.Quat[0], imu.Quat[1]));
    float angularVelocity = theta / sampleRate;
    debug_print("Theta: ");
    debug_println(theta);
    debug_print("Angular velocity (per sec): ");
    debug_println(angularVelocity * 1000000.);

    for (int i = historyLength - 2; i >= 0; i--) {
      angularVelocityWindow[i + 1] = angularVelocityWindow[i];
    }

    angularVelocityWindow[0] = angularVelocity;

    //  Serial.println(imu.algorithmStatus, BIN);

    //   Packet format:
    //   >[./|],[./|],[./|],[./|],[float h],[float p],[float r],[float accel x],[accel y],[accel z],[us since last sample]

    bt.print('>');

    for (int i = 0; i < 4; i++) {
      bool fingerPosition = digitalRead(fingerSwitchPins[i]);
      if (fingerPosition == lastFingerPositions[i]) {
        lastFingerStableTimestamps[i] = millis();
      }
      else if (isGesturing || millis() - lastFingerStableTimestamps[i] >= fingerDebounce) {
        lastFingerPositions[i] = fingerPosition;
        lastFingerStableTimestamps[i] = millis();
      }

      bt.print(lastFingerPositions[i] ? '.' : '|');
      bt.print(',');
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

    bt.print(imu.Quat[0] * -1, 4);
    bt.print(',');

    bt.print(imu.Quat[1], 4);
    bt.print(',');

    bt.print(imu.Quat[2], 4);
    bt.print(',');

    bt.print(imu.ax, 4);
    bt.print(',');

    bt.print(imu.ay, 4);
    bt.print(',');

    bt.print(imu.az, 4);
    bt.print(',');

    // Not sure if this is where the delta should be calculated
    bt.println(sampleRate);

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
  debug_println("Buzz!");
  analogWrite(vibePin, strength);
  timeSinceBuzzStarted = 0;
  currentBuzzDuration = duration;
}

float norm(float oldX, float oldY, float newX, float newY) {
  float xDelta = wrappedDelta(oldX, newX);
  float yDelta = wrappedDelta(oldY, newY);

  return sqrt(xDelta * xDelta + yDelta * yDelta);
}

