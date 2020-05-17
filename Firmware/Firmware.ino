/*************
   Somatic data glove firmware
   by Zack Freedman of Voidstar Lab
   (c) Voidstar Lab 2019
   Somatic is licensed Creative Commons 4.0 Attribution Non-Commercial

   Uses code by Paul Joffressen, Kris Winer, Google, and the Arduino project

   Deploy to Teensy 4.0 at max clock speed
*/

#include <math.h>
#include <Wire.h>
#include <TensorFlowLite.h>
#include  "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// The following dumb bullshit just lets us turn off serial debug
#define DEBUG
#define GET_MACRO(_1, _2, NAME, ...) NAME

byte me;

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

// Keep this declaration here so I can use the debug macros in it
#include "imu.h"

#define training_mode
//#define hid_mode

//#define dont_actually_write
// #define debug_angular_velocity
//#define debug_gesture
#define debug_dump_bt_rx
//#define debug_tensorflow

#define bt Serial1
#define btResponseMaxLength 20
char btResponseBuffer[btResponseMaxLength];

// Globals, needed by TFLite
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

unsigned long fingerDebounce = 100;

const byte btRtsPin = 2;
const byte vibePin = 23;
const byte fingerSwitchPins[] = {9, 10, 11, 12};

#define noHandSign 0
#define mouseHandSign 1
#define keebHandSign 10

bool lastFingerPositions[4];
byte handSign = noHandSign;
unsigned long lastFingerStableTimestamps[4] = {0};

float lastBearing[3];
float yaw;
float pitch;

const float velocityThresholdToBegin = 5. / 1000000.;  // In rad/microsecond
const float velocityThresholdToEnd = 1.5 / 1000000.;  // Also in rad/microsecond
const float mouseVelocityThreshold = 0.01 / 1000000.; // I don't plan on changing the units at this point
const int historyLength = 5;
float angularVelocityWindow[historyLength];

#define mouseScale 500.

float bearingAtLastBuzz[2];
const float distanceBetweenBuzzes = 12. / 180. * PI;
elapsedMillis timeSinceBuzzStarted;
unsigned long currentBuzzDuration;

#define interGestureLockout 200  // Milliseconds to ignore velocity after gesturing, to get your hand into the next position
//#define interGestureLockout 0
bool isDrawingGlyph;
elapsedMillis timeSinceLastGesture;
unsigned long lastTimestamp;

IMU imu = IMU();

elapsedMillis timeSinceLastDebugCommandChar;
#define commandLockout 10000

#define gestureConeAngle 2.0 / 3.0 * PI
float gestureBearingZero[2];
#define maxGestureLength 100
float gestureBuffer[100][2];  // These are all normalized to 0.0-1.0, where 0.5, 0.5 is the starting position
float gestureBufferYawMin;
float gestureBufferYawMax;
float gestureBufferPitchMin;
float gestureBufferPitchMax;
float processedGesture[50][2];
unsigned int gestureBufferLength;

bool isFrozen = false;
elapsedMillis timeSinceFreeze;
#define freezeTime 2000

void setup() {
  Serial.begin(115200);
  bt.begin(115200);

  for (int i = 0; i < 3; i++) {
    delay(200);
    if (setUpBluetooth()) break;
  }

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

  // The following wad-o-code is cribbed straight from the TFLite hello_world project
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(modelBin);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  debug_print("Input tensor size: ");
  debug_print(input->dims->size);
  debug_print(" Shape: (");
  for (int i = 0; i < 3; i++) {
    debug_print(input->dims->data[i]);
    if (i < 2) debug_print(", ");
  }
  debug_println(')');

  debug_print("Output tensor size: ");
  debug_print(output->dims->size);
  debug_print(" Shape: (");
  for (int i = 0; i < 3; i++) {
    debug_print(input->dims->data[i]);
    if (i < 2) debug_print(", ");
  }
  debug_println(')');

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  debug_print("Gesture cone angle: ");
  debug_println(gestureConeAngle, 5);

  lastTimestamp = micros();
}

// TODO: Move non-sensor stuff out of sensor-dependent if block

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
    yaw = imu.Quat[0] * -1;
    pitch = imu.Quat[1];

    float theta = 0;

    theta = asin(norm(lastBearing[0], lastBearing[1], yaw, pitch));
    float angularVelocity = theta / sampleRate;

#ifdef debug_angular_velocity
    debug_print("Theta: ");
    debug_println(theta);
    debug_print("Angular velocity (per sec): ");
    debug_println(angularVelocity * 1000000.);
#endif

    float angularVelocityAverage = angularVelocity;

    for (int i = historyLength - 2; i >= 0; i--) {
      angularVelocityAverage += angularVelocityWindow[i];
      angularVelocityWindow[i + 1] = angularVelocityWindow[i];
    }

    angularVelocityWindow[0] = angularVelocity;

    angularVelocityAverage /= historyLength;

    for (int i = 0; i < 4; i++) {
      bool fingerPosition = digitalRead(fingerSwitchPins[i]);
      if (fingerPosition == lastFingerPositions[i]) {
        lastFingerStableTimestamps[i] = millis();
      }
      else if (millis() - lastFingerStableTimestamps[i] >= fingerDebounce) {
        lastFingerPositions[i] = fingerPosition;
        lastFingerStableTimestamps[i] = millis();
      }
    }

    // Keyboard stuff
    bool handIsMovingFastEnoughToDraw = false;

    if (timeSinceLastGesture < interGestureLockout)
      handIsMovingFastEnoughToDraw = false;
    else if ((isDrawingGlyph && angularVelocityAverage >= velocityThresholdToEnd)
             //        || (!isDrawingGlyph && angularVelocityAverage >= velocityThresholdToBegin)) {
             || (!isDrawingGlyph && angularVelocity >= velocityThresholdToBegin)) {  // Don't use average - we want a snappier start
      handIsMovingFastEnoughToDraw = true;
    }

    if (lastFingerPositions[0] && lastFingerPositions[1] && lastFingerPositions[2] && !lastFingerPositions[3]) {
      handSign = keebHandSign;
    }
    else if (lastFingerPositions[0] && lastFingerPositions[1] && !lastFingerPositions[2] && !lastFingerPositions[3]) {
      handSign = mouseHandSign;
    }
    else
      handSign = noHandSign;

    if (handIsMovingFastEnoughToDraw) {
      if (handSign == keebHandSign) {
        if (!isDrawingGlyph) {
          buzzFor(250, 15);
          bearingAtLastBuzz[0] = yaw;
          bearingAtLastBuzz[1] = pitch;
          gestureBufferLength = 0;
          debug_print("Started gesturing after ");
          debug_print(timeSinceLastGesture);
          debug_println(" ms");
        }
        isDrawingGlyph = true;
      }
    }
    else {
      if (isDrawingGlyph)
        debug_println("Stopped gesturing");
      isDrawingGlyph = false;
    }

    // Record and process gestures! This is the business end!
    char winningGlyph = 0x00;
    float topScore = 0.;

    if (isDrawingGlyph) {
      if (gestureBufferLength < maxGestureLength) {
        if (gestureBufferLength == 0) {
          gestureBearingZero[0] = yaw;
          gestureBearingZero[1] = pitch;
          gestureBuffer[0][0] = gestureBufferYawMin = gestureBufferYawMax = 0.5;
          gestureBuffer[0][1] = gestureBufferPitchMin = gestureBufferPitchMax = 0.5;

#ifdef debug_gesture
          debug_println("Starting gesture!");
          debug_print("Bearing zero is (");
          debug_print(gestureBearingZero[0], 3);
          debug_print(", ");
          debug_print(gestureBearingZero[1], 3);
          debug_println(')');
#endif
        }
        else {
#ifdef debug_gesture
          debug_print("Raw bearing: (");
          debug_print(yaw, 3);
          debug_print(", ");
          debug_print(pitch, 3);
          debug_println(')');
#endif

          float processedYaw = wrappedDelta(gestureBearingZero[0], yaw);
          float processedPitch = wrappedDelta(gestureBearingZero[1], pitch);

          processedYaw = constrain(processedYaw, -0.5 * gestureConeAngle, 0.5 * gestureConeAngle);
          processedPitch = constrain(processedPitch, -0.5 * gestureConeAngle, 0.5 * gestureConeAngle);

          processedYaw /= gestureConeAngle;
          processedPitch /= gestureConeAngle;

          processedYaw += 0.5;
          processedPitch += 0.5;

          gestureBufferYawMin = min(gestureBufferYawMin, processedYaw);
          gestureBufferYawMax = max(gestureBufferYawMax, processedYaw);
          gestureBufferPitchMin = min(gestureBufferPitchMin, processedPitch);
          gestureBufferPitchMax = max(gestureBufferPitchMax, processedPitch);

          gestureBuffer[gestureBufferLength][0] = processedYaw;
          gestureBuffer[gestureBufferLength][1] = processedPitch;

#ifdef debug_gesture
          debug_print("Processed bearing: (");
          debug_print(processedYaw, 3);
          debug_print(", ");
          debug_print(processedPitch, 3);
          debug_println(')');
#endif
        }

        gestureBufferLength++;
      }
#ifdef debug_gesture
      else {
        debug_println("This gesture is too damn long!");
      }
#endif
    }
    else if (gestureBufferLength > 0) {
      timeSinceLastGesture = 0;
      buzzFor(250, 40);

#ifdef debug_gesture
      debug_print("Gesture over - ");
      debug_print(gestureBufferLength);
      debug_println(" points");
#endif

      processBearings(gestureBuffer, gestureBufferLength,
                      gestureBufferYawMin, gestureBufferYawMax, gestureBufferPitchMin, gestureBufferPitchMax,
                      processedGesture, 50);
#ifdef debug_gesture
      for (int i = 0; i < 50; i++) {
        debug_print("Point ");
        debug_print(i);
        debug_print(": (");
        debug_print(processedGesture[i][0]);
        debug_print(", ");
        debug_print(processedGesture[i][1]);
        debug_println(')');
      }
#endif

#ifdef debug_tensorflow
      unsigned long tfBenchmark = micros();
#endif

      // Machine learning hijinks follows
      for (int i = 0; i < 100; i++) {
        input->data.f[i] = processedGesture[i / 2][i % 2];
      }

      TfLiteStatus invokeStatus = interpreter->Invoke();

      if (invokeStatus != kTfLiteOk) {
        error_reporter->Report("Failed to invoke\n");
      }
      else {
        for (int i = 0; i <= charCount; i++) {
          float score = output->data.f[i];
          if (score > topScore) {
            winningGlyph = charMap[i];
            topScore = score;
          }

#ifdef debug_tensorflow
          debug_print("Output tensor ");
          debug_print(i);
          debug_print(" (");

          if (charMap[i] >= ' ')
            debug_print(char(charMap[i]));
          else {
            debug_print("0x");
            debug_print(charMap[i], HEX);
          }

          debug_print("): ");
          debug_println(score, 4);
#endif
        }

#ifdef debug_tensorflow
        debug_print("Inference took ");
        debug_print(micros() - tfBenchmark);
        debug_println("us");
#endif

        for (int i = 0; i < 50; i++) {
          if (winningGlyph >= ' ')
            debug_print(winningGlyph);
          else {
            debug_print("0x");
            debug_print(winningGlyph, HEX);
            debug_print(' ');
          }
        }
        debug_println();
      }

      gestureBufferLength = 0;
    }  // gestureBufferLength > 0

    if (timeSinceBuzzStarted >= 80
        && isDrawingGlyph
        && norm(bearingAtLastBuzz[0], bearingAtLastBuzz[1], yaw, pitch) >= distanceBetweenBuzzes) {
      buzzFor(250, 15);
      bearingAtLastBuzz[0] = yaw;
      bearingAtLastBuzz[1] = pitch;
    }

    if (timeSinceLastDebugCommandChar >= commandLockout) {
      if (!digitalRead(btRtsPin)) {
#ifdef training_mode
        //   Packet format:
        //   >[./|],[./|],[./|],[./|],[float h],[float p],[float r],[float accel x],[accel y],[accel z],[us since last sample]
        char outgoingPacket[100] = {0};
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

        dtostrf(yaw, 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(pitch, 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.Quat[2], 6, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.ax, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.ay, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        dtostrf(imu.az, 7, 4, &outgoingPacket[strlen(outgoingPacket)]);
        outgoingPacket[strlen(outgoingPacket)] = ',';

        itoa(sampleRate, &outgoingPacket[strlen(outgoingPacket)], 10);
        outgoingPacket[strlen(outgoingPacket)] = '\n';

        bt.write(outgoingPacket, strlen(outgoingPacket));
#endif  // ifdef training_mode

#ifdef hid_mode
        byte packetLength = 0;

        if (winningGlyph == 0x00) {
          // Mouse stuff
          if (handSign == mouseHandSign && angularVelocityAverage >= mouseVelocityThreshold) {
            int xStop = wrappedDelta(lastBearing[0], yaw) * mouseScale;
            int yStop = wrappedDelta(lastBearing[1], pitch) * mouseScale;

            if (xStop != 0 || yStop != 0) {
              debug_print("Moving mouse ");
              debug_print(xStop);
              debug_print(" units right and ");
              debug_print(yStop);
              debug_println(" units up");

              // Raw mouse report format. See RN42 HID User Guide, or really the BT spec itself
              byte buf[] = {0xfd, 0x05, 0x02, 0x00, char(xStop), char(yStop), 0x00};
              bt.write(buf, 7);
            }
          }
        }
        else {
          bt.print(winningGlyph);
          debug_print("Typing ");
          debug_println(char(winningGlyph));
        }
#endif  // ifdef hid_mode

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
    lastBearing[0] = yaw;
    lastBearing[1] = pitch;
  }
}

void quaternionMultiply(float* q1, float* q2, float* out) {
  out[0] =  q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
  out[1] = -q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
  out[2] =  q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3] + q1[3] * q2[2];
  out[3] = -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3];
}

bool setUpBluetooth() {
  // Note that this will ALWAYS FAIL when the module is connected in PAIR mode and the CPU alone is reset.
  // I did that for convenience, to use the module's built-in autoconnect.
  // TODO: Connect programmatically

  debug_println("Configuring BT. Emptying buffer...");
  while (bt.available()) bt.read();  // %REBOOT\r message in particular may jam up the works

  debug_println("Entering command mode...");
  bt.print("$$$");
  if (!btExpect("CMD", 200)) {
    debug_println("Failed to enter command mode");
    bt.setTimeout(20);
    return false;
  }

  debug_println("Setting pairing mode");
  bt.println("SM,6");
  if (!btExpect("AOK", 200)) return false;

  // TODO: Check if a setting is already correct, spare the EEPROM some pain

#ifdef training_mode
  debug_println("Switching to SPP mode");
  bt.println("S~,0");
  if (!btExpect("AOK", 200)) return false;
#endif  // training_mode

#ifdef hid_mode
  debug_println("Switching to HID mode");
  bt.println("S~,6");
  if (!btExpect("AOK", 200)) return false;

  debug_println("Configuring HID");
  bt.println("SH,0230");  // 'Combo' descriptor, send output reports, 1 paired device
  if (!btExpect("AOK", 200)) return false;
#endif  // hid_mode

  debug_println("Configuration complete, rebooting");
  bt.println("R,1");
  if (!btExpect("Reboot!", 200)) {
    debug_println("Failed to run reboot command");
    bt.setTimeout(20);
    return false;
  }
  //  else {
  //    btModuleResponse = bt.readStringUntil('\n');
  //    if (btModuleResponse.indexOf("%REBOOT") == -1) {
  //      debug_println("Failed to actually reboot");
  //      bt.setTimeout(20);
  //      return false;
  //    }
  //  }
  debug_println("Rebooted");

  //#ifdef hid_mode
  //  // HID mode requires us to manually reconnect to host device.
  //  // TODO: Do this regularly?
  //  debug_println("Re-entering command mode...");
  //  delay(1000);
  //  bt.print("$$$");
  //  bt.setTimeout(2000);
  //  btModuleResponse = bt.readStringUntil('\n');
  //  if (btModuleResponse.indexOf("CMD") == -1) {
  //    debug_println("Failed to re-enter command mode");
  //    bt.setTimeout(20);
  //    return false;
  //  }
  //
  //  debug_println("Connecting...");
  //  bt.println('C');
  //  btModuleResponse = bt.readStringUntil('\n');
  //  if (btModuleResponse.indexOf("TRYING") == -1) {
  //    debug_println("Failed to run connect command");
  //    bt.setTimeout(20);
  //    return false;
  //  }
  //  bt.setTimeout(10000);
  //  btModuleResponse = bt.readStringUntil('\n');
  //  debug_print("Connection result: ");
  //  debug_println(btModuleResponse);
  //
  //  // TODO: Check if we're already connected
  //  bt.println("---");
  //  btModuleResponse = bt.readStringUntil('\n');
  //  if (btModuleResponse.indexOf("END") == -1) {
  //    debug_println("Failed to leave command mode");
  //    bt.setTimeout(20);
  //    return false;
  //  }
  //#endif  // ifdef hid_mode

  debug_println("Setup complete");
  bt.setTimeout(20);
  return true;
}

bool readBtLine(char * buffer, int bufferLength, char terminator, int timeout) {
  bool done = false;
  unsigned long lastRxTime = millis();

  debug_print("Reading line: ");

  for (int i = 0; i < bufferLength; i++) {
    if (done) buffer[i] = 0;
    else {
      while (!bt.available() && millis() - lastRxTime < timeout) {}
      if (millis() - lastRxTime >= timeout) {
        debug_println("Timed out");
        return false;
      }

      int incoming = bt.read();

      if (incoming == terminator) {
        debug_println("<EOL>");
        done = true;
        buffer[i] = 0;
      }
      else {
        debug_print(char(incoming));
        buffer[i] = incoming;
        lastRxTime = millis();
      }
    }
  }

  return done;
}

bool btExpect(char * target, int timeout) {
  int targetLength = strlen(target);

  debug_print("Expecting '");
  debug_print(target);
  debug_print("' of length ");
  debug_println(targetLength);

  if (!readBtLine(btResponseBuffer, btResponseMaxLength, '\n', timeout)) {
    debug_println("Didn't receive a response line");
    return false;
  }

  debug_print("Response was ");
  debug_println(btResponseBuffer);

  for (int i = 0; i < strlen(btResponseBuffer) - targetLength + 1; i++) {
    for (int j = 0; j < targetLength; j++) {
      if (btResponseBuffer[i + j] != target[j]) break;
      if (j == targetLength - 1) return true;
    }
  }

  return false;
}

/*
  bool expectBtOk() {
  btModuleResponse = bt.readStringUntil('\n');
  debug_println(btModuleResponse);
  if (btModuleResponse.indexOf("OK") == -1) {
    debug_println("No response... command failed");
    bt.setTimeout(20);
    return false;
  }
  debug_println("Command succeeded");
  return true;
  }
*/

float wrappedDelta(float oldValue, float newValue) {
  float delta = oldValue - newValue;

  if (delta > PI)
    return delta - (2. * PI);
  else if (delta < -PI)
    return delta + (2. * PI);
  else
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

void processBearings(float input[][2], unsigned int inputLength, float inputYawMin, float inputYawMax, float inputPitchMin, float inputPitchMax, float output[][2], unsigned int outputLength) {
  unsigned long benchmark = micros();

  float workingBuffer1[maxGestureLength][2] = {0};
  unsigned int buffer1Position = 0;

  // Strip redundant bearings
  workingBuffer1[0][0] = input[0][0];
  workingBuffer1[0][1] = input[0][1];
  buffer1Position = 1;

  for (int i = 1; i < inputLength; i++) {
    float yawDelta = fabs(input[i][0] - input[i - 1][0]);
    float pitchDelta = fabs(input[i][1] - input[i - 1][1]);
    if (yawDelta > 0.000001 || pitchDelta > 0.000001) {
      workingBuffer1[buffer1Position][0] = input[i][0];
      workingBuffer1[buffer1Position][1] = input[i][1];
      buffer1Position++;
    }
#ifdef debug_gesture
    else {
      debug_print("Discarding redundant sample (");
      debug_print(input[i][0], 3);
      debug_print(", ");
      debug_print(input[i][1], 3);
      debug_print(") - too close to previous sample (");
      debug_print(input[i - 1][0], 3);
      debug_print(", ");
      debug_print(input[i - 1][1], 3);
      debug_print("). Yaw delta: ");
      debug_print(yawDelta, 10);
      debug_print(" Pitch delta: ");
      debug_println(pitchDelta, 10);
    }
#endif
  }

  // Trim slop from beginning and end of gesture
  // TODO: Just fail to collect the slop if possible, this looks expensive
  float slopTrimLength = norm(inputYawMin, inputPitchMin, inputYawMax, inputPitchMax) / 10.;

#ifdef debug_gesture
  debug_print("Yaw min: ");
  debug_print(inputYawMin, 3);
  debug_print(" Pitch min: ");
  debug_print(inputPitchMin, 3);
  debug_print(" Yaw max: ");
  debug_print(inputYawMax, 3);
  debug_print(" Pitch max: ");
  debug_print(inputPitchMax, 3);
  debug_print(" Trim length: ");
  debug_println(slopTrimLength, 3);
#endif

  int leadingPointStrippingEnd = 0;
  int trailingPointStrippingStart = buffer1Position - 1;

  for (int i = 1; i < buffer1Position; i++) {
    float distance = norm(workingBuffer1[i][0], workingBuffer1[i][1], workingBuffer1[0][0], workingBuffer1[0][1]);
    if (distance >= slopTrimLength) {
#ifdef debug_gesture
      debug_println("Done counting leading slop");
#endif
      break;
    }
    else {
#ifdef debug_gesture
      debug_print("Stripping leading point (");
      debug_print(workingBuffer1[i][0], 3);
      debug_print(", ");
      debug_print(workingBuffer1[i][1], 3);
      debug_print(") - too close to start point (");
      debug_print(workingBuffer1[0][0], 3);
      debug_print(", ");
      debug_print(workingBuffer1[0][1], 3);
      debug_print("). Must be ");
      debug_print(slopTrimLength, 3);
      debug_print(" but is ");
      debug_println(distance, 3);
#endif

      leadingPointStrippingEnd++;
    }
  }

  for (int i = buffer1Position - 2; i > 0; i--) {
    float distance = norm(workingBuffer1[buffer1Position - 1][0], workingBuffer1[buffer1Position - 1][1],
                          workingBuffer1[i][0], workingBuffer1[i][1]);
    if (distance > slopTrimLength) {
#ifdef debug_gesture
      debug_println("Done counting trailing slop");
#endif
      break;
    }
    else {
#ifdef debug_gesture
      debug_print("Stripping trailing point (");
      debug_print(workingBuffer1[i][0], 3);
      debug_print(", ");
      debug_print(workingBuffer1[i][1], 3);
      debug_print(") - too close to end point (");
      debug_print(workingBuffer1[buffer1Position - 1][0], 3);
      debug_print(", ");
      debug_print(workingBuffer1[buffer1Position - 1][1], 3);
      debug_print("). Must be ");
      debug_print(slopTrimLength, 3);
      debug_print(" but is ");
      debug_println(distance, 3);
#endif

      trailingPointStrippingStart--;
    }
  }

  // While performing the trim, also calculate cumulative lengths
  float workingBuffer2[maxGestureLength][2] = {0};
  unsigned int buffer2Position = 0;
  float segmentCumulativeLengths[maxGestureLength] = {0};
  float curveLength = 0;

  for (int i = 0; i < buffer1Position; i++) {
    if (i == 0 || i == buffer1Position - 1 || (i > leadingPointStrippingEnd && i < trailingPointStrippingStart)) {
      workingBuffer2[buffer2Position][0] = workingBuffer1[i][0];
      workingBuffer2[buffer2Position][1] = workingBuffer1[i][1];

      if (buffer2Position == 0)
        segmentCumulativeLengths[0] = 0;
      else {
        float segmentLength = norm(workingBuffer2[buffer2Position - 1][0], workingBuffer2[buffer2Position - 1][1],
                                   workingBuffer2[buffer2Position][0], workingBuffer2[buffer2Position][1]);

        segmentCumulativeLengths[buffer2Position] = segmentCumulativeLengths[buffer2Position - 1] + segmentLength;

#ifdef debug_gesture
        debug_print("Segment ending in point ");
        debug_print(buffer2Position);
        debug_print(" length: ");
        debug_print(segmentLength, 3);
        debug_print(" Cumul: ");
        debug_println(segmentCumulativeLengths[buffer2Position]);
#endif

        curveLength += segmentLength;
      }

      buffer2Position++;
    }
  }

  // Interpolate and scale
  float targetSegmentLength = curveLength / (outputLength - 1);

#ifdef debug_gesture
  debug_print("Segment length: ");
  debug_println(targetSegmentLength, 3);
#endif

  int firstLongerSample = 0;
  float highPoint[2];
  float lowPoint[2];
  float longDimensionLength = max(inputYawMax - inputYawMin, inputPitchMax - inputPitchMin);

  output[0][0] = map(workingBuffer2[0][0] - inputYawMin, 0., longDimensionLength, 0., 1.);
  output[0][1] = map(workingBuffer2[0][1] - inputPitchMin, 0., longDimensionLength, 0., 1.);

  for (int i = 1; i < outputLength; i++) {
    float targetLength = i * targetSegmentLength;

#ifdef debug_gesture
    debug_print("Looking to place a point at ");
    debug_print(targetLength, 3);
    debug_println(" along curve");
#endif

    if (segmentCumulativeLengths[firstLongerSample] > targetLength) {
#ifdef debug_gesture
      debug_println("Last segment still works");
#endif
    }
    else {
      while (segmentCumulativeLengths[firstLongerSample] < targetLength
             && fabs(segmentCumulativeLengths[firstLongerSample] - targetLength) > 0.00001) {
        firstLongerSample++;

        if (firstLongerSample >= buffer2Position) {
          debug_println("ERROR! Entire line isn't long enough?!");
          return;
        }

        lowPoint[0] = workingBuffer2[firstLongerSample - 1][0];
        lowPoint[1] = workingBuffer2[firstLongerSample - 1][1];
        highPoint[0] = workingBuffer2[firstLongerSample][0];
        highPoint[1] = workingBuffer2[firstLongerSample][1];
      }
    }

    float positionAlongSegment =
      (targetLength - segmentCumulativeLengths[firstLongerSample - 1]) /
      (segmentCumulativeLengths[firstLongerSample] - segmentCumulativeLengths[firstLongerSample - 1]);

    float standardizedYaw = lowPoint[0] + positionAlongSegment * (highPoint[0] - lowPoint[0]);
    float standardizedPitch = lowPoint[1] + positionAlongSegment * (highPoint[1] - lowPoint[1]);

#ifdef debug_gesture
    debug_print("Placed point ");
    debug_print(targetLength - segmentCumulativeLengths[firstLongerSample - 1], 3);
    debug_print(" units (");
    debug_print(positionAlongSegment * 100.0, 0);
    debug_print("%) along the ");
    debug_print(segmentCumulativeLengths[firstLongerSample] - segmentCumulativeLengths[firstLongerSample - 1], 2);
    debug_print(" line between (");
    debug_print(lowPoint[0], 3);
    debug_print(", ");
    debug_print(lowPoint[1], 3);
    debug_print(") and (");
    debug_print(highPoint[0], 3);
    debug_print(", ");
    debug_print(highPoint[1], 3);
    debug_print(") --> (");
    debug_print(standardizedYaw, 3);
    debug_print(", ");
    debug_print(standardizedPitch, 3);
    debug_println(")");
#endif

    // Scale and move point
    standardizedYaw -= inputYawMin;
    standardizedPitch -= inputPitchMin;

    standardizedYaw = map(standardizedYaw, 0., longDimensionLength, 0., 1.);
    standardizedPitch = map(standardizedPitch, 0., longDimensionLength, 0., 1.);

    output[i][0] = standardizedYaw;
    output[i][1] = standardizedPitch;
  }

#ifdef debug_gesture
  debug_print("Wowzers. That took ");
  debug_print(micros() - benchmark);
  debug_println(" usec");
#endif
}
