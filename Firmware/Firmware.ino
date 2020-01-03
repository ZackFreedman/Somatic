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

// End dumb debugging bullshit

//#include <i2c_t3.h>
#include <Wire.h>
#include "imu.h"

byte fingerSwitchPins[] = {9, 10, 11, 12};

unsigned long lastTimestamp;

IMU imu = IMU();

void setup() {
  //  Wire.begin(I2C_MASTER, 0x00, I2C_PINS_18_19, I2C_PULLUP_INT, I2C_RATE_400);
  Serial.begin(115200);

  // Some butt-head didn't leave room for i2c pullups on the lil board.
  // Using adjacent pins as makeshift 3V3 sources :|
  pinMode(17, OUTPUT);
  digitalWrite(17, HIGH);
  pinMode(20, OUTPUT);
  digitalWrite(20, HIGH);

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

  imu.poll();

  // Packet format:
  // >[./|],[./|],[./|],[./|],[float x],[float y],[float z],[float w],[us since last sample]
  
  Serial.print('>');

  for (int i = 0; i < 4; i++) {
    Serial.print(digitalRead(fingerSwitchPins[i]) ? '.' : '|');
    Serial.print(',');
  }

  for (int i = 0; i < 4; i++) {
    Serial.print(imu.Quat[i]);
    Serial.print(',');
  }

  // Not sure if this is where the delta should be calculated
  Serial.println(timestamp - lastTimestamp);

  delay(100);

  lastTimestamp = timestamp;
}
