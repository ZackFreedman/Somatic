#ifndef IMU_h_
#define IMU_h_

//#include <i2c_t3.h>
//#include <Wire.h>
#include "IMU_data.h"

class IMU {
  private:
    // HPR output
    const byte sixAxisOutput = 0b00001000;
    const byte nineAxisOutput = 0b00000000;
    const byte hprOutput = 0b00000100;
    const byte quaternionOutput = 0b00000000;
    const byte rawDataOutput = 0b00000010;
    const byte calibratedDataOutput = 0b00000000;
    const byte standbyEnabled = 0b00000001;
    const byte standbyDisabled = 0b00000000;
    const byte algorithmControlDefaults = quaternionOutput + rawDataOutput + standbyDisabled + sixAxisOutput;
//    const byte algorithmControlDefaults = 0b00000100;

    struct accelerometerCalibrationData
    {
      int16_t accZero_max[3];
      int16_t accZero_min[3];
    };

    struct warmStartData
    {
      uint8_t Sen_param[35][4];
    };

    // I2C EEPROM wrappers. Needs special methods because addresses are two bytes.
    byte M24512DFMreadByte(uint8_t device_address, uint8_t data_address1, uint8_t data_address2);
    void M24512DFMreadBytes(uint8_t device_address, uint8_t data_address1, uint8_t data_address2, uint8_t count, uint8_t * dest);

    byte readByte(uint8_t, uint8_t);
    void readBytes(uint8_t, uint8_t, uint8_t, uint8_t*);
    void writeByte(uint8_t, uint8_t, uint8_t);

    float uint32_reg_to_float(uint8_t*);
    void float_to_bytes(float, uint8_t*);

    void enterPassthrough();
    void leavePassthrough();

    accelerometerCalibrationData loadAccelerometerCalibrationData();
    void calibrateAccelerometer(accelerometerCalibrationData global_conf);

    warmStartData loadWarmStartData();
    void warmStart(warmStartData WS_params);

    void EM7180_set_gyro_FS(uint16_t);
    void EM7180_set_mag_acc_FS(uint16_t, uint16_t);
    void EM7180_set_integer_param(uint8_t, uint32_t);
    void EM7180_set_float_param(uint8_t, float);

    void readSENtralQuatData(float*);
    void readSENtralAccelData(int16_t*);
    void readSENtralGyroData(int16_t*);
    void readSENtralMagData(int16_t*);

    byte param[4];

  public:
    IMU() {}

    void setup();
    void poll();
    void sleep();
    void wake();

    byte algorithmStatus;
    bool faulty;

    float ax, ay, az;
    float gx, gy, gz;
    float mx, my, mz;
    float Quat[4];
    float Yaw, Pitch, Roll;
};

byte IMU::readByte(uint8_t address, uint8_t subAddress)
{
  uint8_t data; // `data` will store the register data
  Wire.beginTransmission(address);         // Initialize the Tx buffer
  Wire.write(subAddress);                   // Put slave register address in Tx buffer
//  Wire.endTransmission(I2C_NOSTOP);        // Send the Tx buffer, but send a restart to keep connection alive
    Wire.endTransmission(false);             // Send the Tx buffer, but send a restart to keep connection alive
  //  Wire.requestFrom(address, 1);  // Read one byte from slave register address
  Wire.requestFrom(address, (size_t) 1);   // Read one byte from slave register address
  data = Wire.read();                      // Fill Rx buffer with result
  return data;                             // Return data read from slave register
}

void IMU::readBytes(uint8_t address, uint8_t subAddress, uint8_t count, uint8_t * dest)
{
  Wire.beginTransmission(address);   // Initialize the Tx buffer
  Wire.write(subAddress);            // Put slave register address in Tx buffer
//  Wire.endTransmission(I2C_NOSTOP);  // Send the Tx buffer, but send a restart to keep connection alive
    Wire.endTransmission(false);       // Send the Tx buffer, but send a restart to keep connection alive
  uint8_t i = 0;
  //        Wire.requestFrom(address, count);  // Read bytes from slave register address
  Wire.requestFrom(address, (size_t) count);  // Read bytes from slave register address
  while (Wire.available()) {
    dest[i++] = Wire.read();
  }         // Put read results in the Rx buffer
}

void IMU::writeByte(uint8_t address, uint8_t subAddress, uint8_t data)
{
  Wire.beginTransmission(address);  // Initialize the Tx buffer
  Wire.write(subAddress);           // Put slave register address in Tx buffer
  Wire.write(data);                 // Put data in Tx buffer
  Wire.endTransmission();           // Send the Tx buffer
}

uint8_t IMU::M24512DFMreadByte(uint8_t device_address, uint8_t data_address1, uint8_t data_address2)
{
  uint8_t data; // `data` will store the register data
  Wire.beginTransmission(device_address);         // Initialize the Tx buffer
  Wire.write(data_address1);                      // Put slave register address in Tx buffer
  Wire.write(data_address2);                      // Put slave register address in Tx buffer
//  Wire.endTransmission(I2C_NOSTOP);               // Send the Tx buffer, but send a restart to keep connection alive
  Wire.endTransmission(false);
  Wire.requestFrom(device_address, (size_t)1);   // Read one byte from slave register address
  data = Wire.read();                             // Fill Rx buffer with result
  return data;                                    // Return data read from slave register
}

void IMU::M24512DFMreadBytes(uint8_t device_address, uint8_t data_address1, uint8_t data_address2, uint8_t count, uint8_t * dest)
{
  Wire.beginTransmission(device_address);            // Initialize the Tx buffer
  Wire.write(data_address1);                         // Put slave register address in Tx buffer
  Wire.write(data_address2);                         // Put slave register address in Tx buffer
//  Wire.endTransmission(I2C_NOSTOP);                  // Send the Tx buffer, but send a restart to keep connection alive
  Wire.endTransmission(false);
  uint8_t i = 0;
  Wire.requestFrom(device_address, (size_t)count);  // Read bytes from slave register address
  while (Wire.available())
  {
    dest[i++] = Wire.read();
  }                                                   // Put read results in the Rx buffer
}

void IMU::enterPassthrough() {
  uint8_t statusByte = 0;

  // First put SENtral in standby mode
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x01);
  delay(5);

  // Place SENtral in pass-through mode
  writeByte(EM7180_ADDRESS, EM7180_PassThruControl, 0x01);
  delay(5);
  statusByte = readByte(EM7180_ADDRESS, EM7180_PassThruStatus);
  while (!(statusByte & 0x01))
  {
    statusByte = readByte(EM7180_ADDRESS, EM7180_PassThruStatus);
    delay(5);
  }
}

void IMU::leavePassthrough() {
  uint8_t statusByte = 0;
  // Cancel pass-through mode
  writeByte(EM7180_ADDRESS, EM7180_PassThruControl, 0x00);
  delay(5);
  statusByte = readByte(EM7180_ADDRESS, EM7180_PassThruStatus);
  while ((statusByte & 0x01))
  {
    statusByte = readByte(EM7180_ADDRESS, EM7180_PassThruStatus);
    delay(5);
  }
  // Re-start algorithm
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults);
  delay(5);
  statusByte = readByte(EM7180_ADDRESS, EM7180_AlgorithmStatus);
  while ((statusByte & 0x01))
  {
    statusByte = readByte(EM7180_ADDRESS, EM7180_AlgorithmStatus);
    delay(5);
  }
}

IMU::accelerometerCalibrationData IMU::loadAccelerometerCalibrationData() {
  // Loads calibration data from EM7180SFP's onboard EEPROM
  // Must be in passthrough mode to access EEPROM!

  accelerometerCalibrationData global_conf;

  uint8_t data[12];
  uint8_t axis;

  M24512DFMreadBytes(M24512DFM_DATA_ADDRESS, 0x7f, 0x8c, 12, data); // Page 255
  for (axis = 0; axis < 3; axis++)
  {
    global_conf.accZero_max[axis] = ((int16_t)(data[(2 * axis + 1)] << 8) | data[2 * axis]);
    global_conf.accZero_min[axis] = ((int16_t)(data[(2 * axis + 7)] << 8) | data[(2 * axis + 6)]);
  }

  return global_conf;
}

// Write calibration data to coprocessor

void IMU::calibrateAccelerometer(accelerometerCalibrationData global_conf) {
  int64_t big_cal_num;
  union
  {
    int16_t cal_num;
    unsigned char cal_num_byte[2];
  };

  big_cal_num = (4096000000 / (global_conf.accZero_max[0] - global_conf.accZero_min[0])) - 1000000;
  cal_num = (int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP36, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP37, cal_num_byte[1]);

  big_cal_num = (4096000000 / (global_conf.accZero_max[1] - global_conf.accZero_min[1])) - 1000000;
  cal_num = (int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP38, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP39, cal_num_byte[1]);

  big_cal_num = (4096000000 / (global_conf.accZero_max[2] - global_conf.accZero_min[2])) - 1000000;
  cal_num = (int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP40, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP50, cal_num_byte[1]);

  big_cal_num = (((2048 - global_conf.accZero_max[0]) + (-2048 - global_conf.accZero_min[0])) * 100000) / 4096;
  cal_num = (int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP51, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP52, cal_num_byte[1]);

  big_cal_num = (((2048 - global_conf.accZero_max[1]) + (-2048 - global_conf.accZero_min[1])) * 100000) / 4096;
  cal_num = (int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP53, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP54, cal_num_byte[1]);

  big_cal_num = (((2048 - global_conf.accZero_max[2]) + (-2048 - global_conf.accZero_min[2])) * 100000) / 4096;
  cal_num = -(int16_t)big_cal_num;
  writeByte(EM7180_ADDRESS, EM7180_GP55, cal_num_byte[0]);
  writeByte(EM7180_ADDRESS, EM7180_GP56, cal_num_byte[1]);
}

IMU::warmStartData IMU::loadWarmStartData() {
  // Reads warm-start params from EM7180SFP onboard EEPROM
  // Must be in passthrough mode to access EEPROM

  warmStartData WS_params;
  uint8_t data[140];
  uint8_t paramnum;
  M24512DFMreadBytes(M24512DFM_DATA_ADDRESS, 0x7f, 0x80, 12, &data[128]); // Page 255
  delay(100);
  M24512DFMreadBytes(M24512DFM_DATA_ADDRESS, 0x7f, 0x00, 128, &data[0]); // Page 254
  for (paramnum = 0; paramnum < 35; paramnum++) // 35 parameters
  {
    for (uint8_t i = 0; i < 4; i++)
    {
      WS_params.Sen_param[paramnum][i] = data[(paramnum * 4 + i)];
    }
  }

  return WS_params;
}

void IMU::warmStart(warmStartData WS_params) {
  uint8_t param = 1;
  uint8_t STAT;

  // Parameter is the decimal value with the MSB set high to indicate a paramter write processs
  param = param | 0x80;
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, WS_params.Sen_param[0][0]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, WS_params.Sen_param[0][1]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, WS_params.Sen_param[0][2]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, WS_params.Sen_param[0][3]);
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, param);

  // Request parameter transfer procedure
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x80);

  // Check the parameter acknowledge register and loop until the result matches parameter request byte
  STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  while (!(STAT == param))
  {
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  }
  for (uint8_t i = 1; i < 35; i++)
  {
    param = (i + 1) | 0x80;
    writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, WS_params.Sen_param[i][0]);
    writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, WS_params.Sen_param[i][1]);
    writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, WS_params.Sen_param[i][2]);
    writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, WS_params.Sen_param[i][3]);
    writeByte(EM7180_ADDRESS, EM7180_ParamRequest, param);

    // Check the parameter acknowledge register and loop until the result matches parameter request byte
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
    while (!(STAT == param))
    {
      STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
    }
  }
  // Parameter request = 0 to end parameter transfer process
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0x00);
}

void IMU::EM7180_set_gyro_FS (uint16_t gyro_fs) {
  uint8_t bytes[4], STAT;
  bytes[0] = gyro_fs & (0xFF);
  bytes[1] = (gyro_fs >> 8) & (0xFF);
  bytes[2] = 0x00;
  bytes[3] = 0x00;
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, bytes[0]); //Gyro LSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, bytes[1]); //Gyro MSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, bytes[2]); //Unused
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, bytes[3]); //Unused
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0xCB); //Parameter 75; 0xCB is 75 decimal with the MSB set high to indicate a paramter write processs
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x80); //Request parameter transfer procedure
  STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge); //Check the parameter acknowledge register and loop until the result matches parameter request byte
  while (!(STAT == 0xCB)) {
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  }
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0x00); //Parameter request = 0 to end parameter transfer process
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults); // Re-start algorithm
}

void IMU::EM7180_set_mag_acc_FS (uint16_t mag_fs, uint16_t acc_fs) {
  uint8_t bytes[4], STAT;
  bytes[0] = mag_fs & (0xFF);
  bytes[1] = (mag_fs >> 8) & (0xFF);
  bytes[2] = acc_fs & (0xFF);
  bytes[3] = (acc_fs >> 8) & (0xFF);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, bytes[0]); //Mag LSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, bytes[1]); //Mag MSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, bytes[2]); //Acc LSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, bytes[3]); //Acc MSB
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0xCA); //Parameter 74; 0xCA is 74 decimal with the MSB set high to indicate a paramter write processs
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x80); //Request parameter transfer procedure
  STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge); //Check the parameter acknowledge register and loop until the result matches parameter request byte
  while (!(STAT == 0xCA)) {
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  }
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0x00); //Parameter request = 0 to end parameter transfer process
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults); // Re-start algorithm
}

float IMU::uint32_reg_to_float (uint8_t *buf) {
  union {
    uint32_t ui32;
    float f;
  } u;

  u.ui32 =     (((uint32_t)buf[0]) +
                (((uint32_t)buf[1]) <<  8) +
                (((uint32_t)buf[2]) << 16) +
                (((uint32_t)buf[3]) << 24));
  return u.f;
}


void IMU::float_to_bytes (float param_val, uint8_t *buf) {
  union {
    float f;
    uint8_t comp[sizeof(float)];
  } u;
  u.f = param_val;
  for (uint8_t i = 0; i < sizeof(float); i++) {
    buf[i] = u.comp[i];
  }
  //Convert to LITTLE ENDIAN
  for (uint8_t i = 0; i < sizeof(float); i++) {
    buf[i] = buf[(sizeof(float) - 1) - i];
  }
}

void IMU::EM7180_set_integer_param (uint8_t param, uint32_t param_val) {
  uint8_t bytes[4], STAT;
  bytes[0] = param_val & (0xFF);
  bytes[1] = (param_val >> 8) & (0xFF);
  bytes[2] = (param_val >> 16) & (0xFF);
  bytes[3] = (param_val >> 24) & (0xFF);
  param = param | 0x80; //Parameter is the decimal value with the MSB set high to indicate a paramter write processs
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, bytes[0]); //Param LSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, bytes[1]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, bytes[2]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, bytes[3]); //Param MSB
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, param);
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x80); //Request parameter transfer procedure
  STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge); //Check the parameter acknowledge register and loop until the result matches parameter request byte
  while (!(STAT == param)) {
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  }
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0x00); //Parameter request = 0 to end parameter transfer process
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults); // Re-start algorithm
}

void IMU::EM7180_set_float_param (uint8_t param, float param_val) {
  uint8_t bytes[4], STAT;
  float_to_bytes (param_val, &bytes[0]);
  param = param | 0x80; //Parameter is the decimal value with the MSB set high to indicate a paramter write processs
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte0, bytes[0]); //Param LSB
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte1, bytes[1]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte2, bytes[2]);
  writeByte(EM7180_ADDRESS, EM7180_LoadParamByte3, bytes[3]); //Param MSB
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, param);
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, 0x80); //Request parameter transfer procedure
  STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge); //Check the parameter acknowledge register and loop until the result matches parameter request byte
  while (!(STAT == param)) {
    STAT = readByte(EM7180_ADDRESS, EM7180_ParamAcknowledge);
  }
  writeByte(EM7180_ADDRESS, EM7180_ParamRequest, 0x00); //Parameter request = 0 to end parameter transfer process
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults); // Re-start algorithm
}

void IMU::readSENtralQuatData(float * destination)
{
  uint8_t rawData[16];  // x/y/z quaternion register data stored here
  readBytes(EM7180_ADDRESS, EM7180_QX, 16, &rawData[0]);       // Read the sixteen raw data registers into data array
  destination[0] = uint32_reg_to_float (&rawData[0]);
  destination[1] = uint32_reg_to_float (&rawData[4]);
  destination[2] = uint32_reg_to_float (&rawData[8]);
  destination[3] = uint32_reg_to_float (&rawData[12]);
}

void IMU::readSENtralAccelData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z accel register data stored here
  readBytes(EM7180_ADDRESS, EM7180_AX, 6, &rawData[0]);       // Read the six raw data registers into data array
  destination[0] = (int16_t) (((int16_t)rawData[1] << 8) | rawData[0]);  // Turn the MSB and LSB into a signed 16-bit value
  destination[1] = (int16_t) (((int16_t)rawData[3] << 8) | rawData[2]);
  destination[2] = (int16_t) (((int16_t)rawData[5] << 8) | rawData[4]);
}

void IMU::readSENtralGyroData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z gyro register data stored here
  readBytes(EM7180_ADDRESS, EM7180_GX, 6, &rawData[0]);  // Read the six raw data registers sequentially into data array
  destination[0] = (int16_t) (((int16_t)rawData[1] << 8) | rawData[0]);   // Turn the MSB and LSB into a signed 16-bit value
  destination[1] = (int16_t) (((int16_t)rawData[3] << 8) | rawData[2]);
  destination[2] = (int16_t) (((int16_t)rawData[5] << 8) | rawData[4]);
}

void IMU::readSENtralMagData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z gyro register data stored here
  readBytes(EM7180_ADDRESS, EM7180_MX, 6, &rawData[0]);  // Read the six raw data registers sequentially into data array
  destination[0] = (int16_t) (((int16_t)rawData[1] << 8) | rawData[0]);   // Turn the MSB and LSB into a signed 16-bit value
  destination[1] = (int16_t) (((int16_t)rawData[3] << 8) | rawData[2]);
  destination[2] = (int16_t) (((int16_t)rawData[5] << 8) | rawData[4]);
}

void IMU::setup()
{
  //  I2Cscan(); // should detect SENtral at 0x28

  // TODO Ping Sentral to see if it's connected
  // TODO handle failures

  // Do a forceful reset in case something funny happened during programming
  writeByte(EM7180_ADDRESS, EM7180_ResetRequest, 1);

  delay(500);

  // Read SENtral device information
  uint16_t ROM1 = readByte(EM7180_ADDRESS, EM7180_ROMVersion1);
  uint16_t ROM2 = readByte(EM7180_ADDRESS, EM7180_ROMVersion2);
  debug_print("EM7180 ROM Version: 0x"); debug_print(ROM1, HEX); debug_println(ROM2, HEX); debug_println("Should be: 0xE609");
  uint16_t RAM1 = readByte(EM7180_ADDRESS, EM7180_RAMVersion1);
  uint16_t RAM2 = readByte(EM7180_ADDRESS, EM7180_RAMVersion2);
  debug_print("EM7180 RAM Version: 0x"); debug_print(RAM1); debug_println(RAM2);
  uint8_t PID = readByte(EM7180_ADDRESS, EM7180_ProductID);
  debug_print("EM7180 ProductID: 0x"); debug_print(PID, HEX); debug_println(" Should be: 0x80");
  uint8_t RID = readByte(EM7180_ADDRESS, EM7180_RevisionID);
  debug_print("EM7180 RevisionID: 0x"); debug_print(RID, HEX); debug_println(" Should be: 0x02");

  //  delay(1000); // give some time to read the screen

  // Check SENtral status, make sure EEPROM upload of firmware was accomplished
  byte STAT = (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x01);
  if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x01)  debug_println("EEPROM detected on the sensor bus!");
  if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x02)  debug_println("EEPROM uploaded config file!");
  if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x04)  debug_println("EEPROM CRC incorrect!");
  if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x08)  debug_println("EM7180 in initialized state!");
  if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x10)  debug_println("No EEPROM detected!");
  int count = 0;
  while (!STAT) {
    writeByte(EM7180_ADDRESS, EM7180_ResetRequest, 0x01);
    delay(500);
    count++;
    STAT = (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x01);
    if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x01)  debug_println("EEPROM detected on the sensor bus!");
    if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x02)  debug_println("EEPROM uploaded config file!");
    if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x04)  debug_println("EEPROM CRC incorrect!");
    if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x08)  debug_println("EM7180 in initialized state!");
    if (readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x10)  debug_println("No EEPROM detected!");
    if (count > 10) break;
  }

  if (!(readByte(EM7180_ADDRESS, EM7180_SentralStatus) & 0x04))  debug_println("EEPROM upload successful!");

  delay(100);

  // Load warm start and accelerometer calibration data
  enterPassthrough();
  accelerometerCalibrationData calibration = loadAccelerometerCalibrationData();
//  warmStartData warmData = loadWarmStartData();
  leavePassthrough();

  // Set up the SENtral as sensor bus in normal operating mode
  // Enter EM7180 initialized state
  writeByte(EM7180_ADDRESS, EM7180_HostControl, 0x00); // set SENtral in initialized state to configure registers
  writeByte(EM7180_ADDRESS, EM7180_PassThruControl, 0x00); // make sure pass through mode is off
  calibrateAccelerometer(calibration);
  writeByte(EM7180_ADDRESS, EM7180_HostControl, 0x01); // Cargo-culted in from example. Labeled 'Force initialize'.
//  warmStart(warmData);
  writeByte(EM7180_ADDRESS, EM7180_HostControl, 0x00); // Complete warm start
  // Set accel/gyro/mag refresh rates
  writeByte(EM7180_ADDRESS, EM7180_QRateDivisor, 0x02); // 100 Hz
  writeByte(EM7180_ADDRESS, EM7180_MagRate, 0x1E); // 30 Hz
  writeByte(EM7180_ADDRESS, EM7180_AccelRate, 0x0A); // 100/10 Hz
  writeByte(EM7180_ADDRESS, EM7180_GyroRate, 0x14); // 200/10 Hz

  // Configure operating mode
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults); // read scale sensor data
  // Enable interrupt to host upon certain events
  // choose interrupts when quaternions updated (0x04), an error occurs (0x02), or the SENtral needs to be reset(0x01)
  writeByte(EM7180_ADDRESS, EM7180_EnableEvents, 0x07);
  // Enable EM7180 run mode
  writeByte(EM7180_ADDRESS, EM7180_HostControl, 0x01); // set SENtral in normal run mode
  delay(100);

  // EM7180 parameter adjustments
  debug_println("Beginning Parameter Adjustments");

  //Enable stillness mode
  EM7180_set_integer_param (0x49, 0x00);

  //Write desired sensor full scale ranges to the EM7180
  EM7180_set_mag_acc_FS (0x3E8, 0x08); // 1000 uT, 8 g
  EM7180_set_gyro_FS (0x7D0); // 2000 dps

  // Read EM7180 status
  uint8_t runStatus = readByte(EM7180_ADDRESS, EM7180_RunStatus);
  if (runStatus & 0x01) debug_println(" EM7180 run status = normal mode");
  uint8_t algoStatus = readByte(EM7180_ADDRESS, EM7180_AlgorithmStatus);
  if (algoStatus & 0x01) debug_println(" EM7180 standby status");
  if (algoStatus & 0x02) debug_println(" EM7180 algorithm slow");
  if (algoStatus & 0x04) debug_println(" EM7180 in stillness mode");
  if (algoStatus & 0x08) debug_println(" EM7180 mag calibration completed");
  if (algoStatus & 0x10) debug_println(" EM7180 magnetic anomaly detected");
  if (algoStatus & 0x20) debug_println(" EM7180 unreliable sensor data");
  uint8_t passthruStatus = readByte(EM7180_ADDRESS, EM7180_PassThruStatus);
  if (passthruStatus & 0x01) debug_print(" EM7180 in passthru mode!");
  uint8_t eventStatus = readByte(EM7180_ADDRESS, EM7180_EventStatus);
  if (eventStatus & 0x01) debug_println(" EM7180 CPU reset");
  if (eventStatus & 0x02) debug_println(" EM7180 Error");
  if (eventStatus & 0x04) debug_println(" EM7180 new quaternion result");
  if (eventStatus & 0x08) debug_println(" EM7180 new mag result");
  if (eventStatus & 0x10) debug_println(" EM7180 new accel result");
  if (eventStatus & 0x20) debug_println(" EM7180 new gyro result");

  //  delay(1000); // give some time to read the screen

  // Check sensor status
  uint8_t sensorStatus = readByte(EM7180_ADDRESS, EM7180_SensorStatus);
  debug_print(" EM7180 sensor status = "); debug_println(sensorStatus);
  if (sensorStatus & 0x01) debug_print("Magnetometer not acknowledging!");
  if (sensorStatus & 0x02) debug_print("Accelerometer not acknowledging!");
  if (sensorStatus & 0x04) debug_print("Gyro not acknowledging!");
  if (sensorStatus & 0x10) debug_print("Magnetometer ID not recognized!");
  if (sensorStatus & 0x20) debug_print("Accelerometer ID not recognized!");
  if (sensorStatus & 0x40) debug_print("Gyro ID not recognized!");

  debug_print("Actual MagRate = "); debug_print(readByte(EM7180_ADDRESS, EM7180_ActualMagRate)); debug_println(" Hz");
  debug_print("Actual AccelRate = "); debug_print(10 * readByte(EM7180_ADDRESS, EM7180_ActualAccelRate)); debug_println(" Hz");
  debug_print("Actual GyroRate = "); debug_print(10 * readByte(EM7180_ADDRESS, EM7180_ActualGyroRate)); debug_println(" Hz");

  //    delay(1000); // give some time to read the screen
}

void IMU::poll() {
  // Check event status register, way to chech data ready by polling rather than interrupt
  uint8_t eventStatus = readByte(EM7180_ADDRESS, EM7180_EventStatus); // reading clears the register

  // Check for errors
  if (eventStatus & 0x02) { // error detected, what is it?

    uint8_t errorStatus = readByte(EM7180_ADDRESS, EM7180_ErrorRegister);
    if (!errorStatus) {
      debug_print(" EM7180 sensor status = "); debug_println(errorStatus);
      if (errorStatus == 0x11) debug_print("Magnetometer failure!");
      if (errorStatus == 0x12) debug_print("Accelerometer failure!");
      if (errorStatus == 0x14) debug_print("Gyro failure!");
      if (errorStatus == 0x21) debug_print("Magnetometer initialization failure!");
      if (errorStatus == 0x22) debug_print("Accelerometer initialization failure!");
      if (errorStatus == 0x24) debug_print("Gyro initialization failure!");
      if (errorStatus == 0x30) debug_print("Math error!");
      if (errorStatus == 0x80) debug_print("Invalid sample rate!");
    }

    // Handle errors ToDo
    faulty = true;

  }
  else faulty = false;

  algorithmStatus = readByte(EM7180_ADDRESS, EM7180_AlgorithmStatus);

  // if no errors, see if new data is ready

  /*
    if (eventStatus & 0x10) { // new acceleration data available
    readSENtralAccelData(accelCount);

    // Now we'll calculate the accleration value into actual g's
    ax = (float)accelCount[0] * 0.000488; // get actual g value
    ay = (float)accelCount[1] * 0.000488;
    az = (float)accelCount[2] * 0.000488;
    }

    if (eventStatus & 0x20) { // new gyro data available
    readSENtralGyroData(gyroCount);

    // Now we'll calculate the gyro value into actual dps's
    gx = (float)gyroCount[0] * 0.153; // get actual dps value
    gy = (float)gyroCount[1] * 0.153;
    gz = (float)gyroCount[2] * 0.153;
    }

    if (eventStatus & 0x08) { // new mag data available
    readSENtralMagData(magCount);

    // Calculate the magnetometer values in milliGauss
    // Temperature-compensated magnetic field is in 32768 LSB/10 microTesla
    mx = (float)magCount[0] * 0.32768; // get actual magnetometer value in mGauss
    my = (float)magCount[1] * 0.32768;
    mz = (float)magCount[2] * 0.32768;
    }
  */

  //    if (readByte(EM7180_ADDRESS, EM7180_EventStatus) & 0x04) { // new quaternion data available
  if (eventStatus & 0x04) { // new quaternion data available
    readSENtralQuatData(Quat);
    /*
    Yaw   = atan2(2.0f * (Quat[0] * Quat[1] + Quat[3] * Quat[2]), Quat[3] * Quat[3] + Quat[0] * Quat[0] - Quat[1] * Quat[1] - Quat[2] * Quat[2]);
    Pitch = -asin(2.0f * (Quat[0] * Quat[2] - Quat[3] * Quat[1]));
    Roll  = atan2(2.0f * (Quat[3] * Quat[0] + Quat[1] * Quat[2]), Quat[3] * Quat[3] - Quat[0] * Quat[0] - Quat[1] * Quat[1] + Quat[2] * Quat[2]);
    Pitch *= 180.0f / PI;
    Yaw   *= 180.0f / PI;
    Yaw   -= 12.9f; // Declination in lower Manhattan
    Roll  *= 180.0f / PI;
    */

    Yaw = (Quat[0] * 180.0f / PI) - 12.9f;
    Pitch = Quat[1] * 180.0f / PI;
    Roll = Quat[2] * 180.0f / PI;
  }
}

void IMU::sleep() {
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults + 0x01); // Bit 0 is StandbyEnable
}

void IMU::wake() {
  writeByte(EM7180_ADDRESS, EM7180_AlgorithmControl, algorithmControlDefaults);
}

#endif
