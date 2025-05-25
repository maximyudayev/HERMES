############
#
# Copyright (c) 2024 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

import clr
import sys

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from WaveplusLab.Shared.Definitions import SensorType # type: ignore
from CyUSB import * # type: ignore


class DeviceStateEnum:
  NOT_CONNECTED = DeviceState.NotConnected # type: ignore # No USB device is connected to the bus
  INITIALIZING = DeviceState.Initializing # type: ignore # WavePlus is initializing
  COMMUNICATION_ERROR = DeviceState.CommunicationError # type: ignore # No communication with WavePlus device available
  INITIALIZING_ERROR = DeviceState.InitializingError # type: ignore # WavePlus device was not correctly initialized
  IDLE = DeviceState.Idle # type: ignore # WavePlus device is ready for use
  CAPTURING = DeviceState.Capturing # type: ignore # WavePlus device is acquiting data


class DeviceErrorEnum:
  SUCCESS = DeviceError.Success # type: ignore
  DEVICE_NOTCONNECTED = DeviceError.DeviceNotConnected # type: ignore
  SENDING_COMMAND = DeviceError.SendingCommand # type: ignore
  RECEIVING_COMMAND_REPLY = DeviceError.ReceivingCommandReply # type: ignore
  DEVICE_ERROR_EXECUTING_COMMAND = DeviceError.DeviceErrorExecutingCommand # type: ignore
  CONFIGURING_CAPTURE = DeviceError.ConfiguringCapture # type: ignore
  WRONG_CAPTURE_CONFIGURATION_IMU_ACQ_TYPE = DeviceError.WrongCaptureConfigurationImuAcqType # type: ignore
  WRONG_CAPTURE_CONFIGURATION_SAMPLING_RATE = DeviceError.WrongCaptureConfigurationSamplingRate # type: ignore
  WRONG_CAPTURE_CONFIGURATION_SAMPLING_RATE_FROM_DEVICE = DeviceError.WrongCaptureConfigurationSamplingRateFromDevice # type: ignore
  WRONG_CAPTURE_CONFIGURATION_DATA_AVAILABLE_EVENT_PERIOD = DeviceError.WrongCaptureConfigurationDataAvailableEventPeriod # type: ignore
  WRONG_SENSOR_CONFIGURATION_ACCELEROMETER_FULL_SCALE = DeviceError.WrongSensorConfigurationAccelerometerFullScale # type: ignore
  WRONG_SENSOR_CONFIGURATION_GYROSCOPE_FULL_SCALE = DeviceError.WrongSensorConfigurationGyroscopeFullScale # type: ignore
  WRONG_SENSOR_CONFIGURATION_SENSOR_TYPE = DeviceError.WrongSensorConfigurationSensorType # type: ignore
  WRONG_FOOT_SW_PROTOCOL = DeviceError.WrongFootSwProtocol # type: ignore
  WRONG_DEVICE_TYPE_FROM_DEVICE = DeviceError.WrongDeviceTypeFromDevice # type: ignore
  WRONG_SENSOR_NUMBER = DeviceError.WrongSensorNumber # type: ignore
  WRONG_FOOT_SW_SENSOR_NUMBER = DeviceError.WrongFootSwSensorNumber # type: ignore
  FOOT_SW_SENSOR_NOT_INSTALLED = DeviceError.FootSwSensorNotInstalled # type: ignore
  READING_BACK_CAPTURE_CONFIGURATION_SAMPLING_RATE = DeviceError.ReadingBackCaptureConfigurationSamplingRate # type: ignore
  READING_BACK_COMMUNICATION_TEST_DATA = DeviceError.ReadingBackCommunicationTestData # type: ignore
  READING_BACK_SENSOR_COMMAND_BUFFER = DeviceError.ReadingBackSensorCommandBuffer # type: ignore
  DATA_TRANSFER_THREADING_STARTING_TIMEOUT = DeviceError.DataTransferThreadStartingTimeout# type: ignore
  ACTION_NOT_ALLOWED_IN_THE_CURRENT_DEVICE_STATE = DeviceError.ActionNotAllowedInTheCurrentDeviceState # type: ignore
  ACTION_NOT_ALLOWED_IN_THE_CURRENT_DATA_TRANSFER_STATE = DeviceError.ActionNotAllowedInTheCurrentDataTransferState # type: ignore
  WRONG_DEVICE_STATE = DeviceError.WrongDeviceState # type: ignore
  WRONG_DEVICE_ACTION = DeviceError.WrongDeviceAction # type: ignore
  TIMEOUT_EXECUTING_SENSOR_COMMAND = DeviceError.TimeoutExecutingSensorCommand # type: ignore
  COMMAND_NOT_EXECUTED_BY_ALL_THE_SENSORS = DeviceError.CommandNotExecutedByAllTheSensors # type: ignore
  WRONG_DAQ_TIMEOUT_VALUE = DeviceError.WrongDaqTimeOutValue # type: ignore
  BAD_SENSOR_COMMUNICATION = DeviceError.BadSensorCommunication # type: ignore
  SYNC_BUFFER1_OVERRUN = DeviceError.SyncBuffer1Overrun # type: ignore
  SYNC_BUFFER2_OVERRUN = DeviceError.SyncBuffer2Overrun # type: ignore
  IMU_CALIBRATION_NOT_AVAILABLE = DeviceError.ImuCalibrationNotAvailable # type: ignore


class DaqDeviceExceptionTypeEnum:
  DEVICE_NOT_CONNECTED = DaqDeviceExceptionType.deviceNotConnected # type: ignore
  UNABLE_TO_START_CAPTURE_DATA_TRANSFER = DaqDeviceExceptionType.unableToStartCaptureDataTransfer # type: ignore
  UNABLE_TO_START_IMPEDANCE_DATA_TRANSFER = DaqDeviceExceptionType.unableToStartImpedanceDataTransfer # type: ignore
  UNABLE_TO_START_CAPTURING = DaqDeviceExceptionType.unableToStartCapturing # type: ignore
  UNABLE_TO_STOP_CAPTURING = DaqDeviceExceptionType.unableToStopCapturing # type: ignore
  UNABLE_TO_READ_SENSOR_MEMORY_STATUS = DaqDeviceExceptionType.unableToReadSensorMemoryStatus # type: ignore
  UNABLE_TO_GET_CAPTURE_CONFIGURATION = DaqDeviceExceptionType.unableToGetCaptureConfiguration # type: ignore
  UNABLE_TO_SET_CAPTURE_CONFIGURATION = DaqDeviceExceptionType.unableToSetCaptureConfiguration # type: ignore
  UNABLE_TO_GET_INSTALLED_SENSORS = DaqDeviceExceptionType.unableToGetInstalledSensors # type: ignore
  UNABLE_TO_GET_DEVICE_TYPE = DaqDeviceExceptionType.unableToGetDeviceType # type: ignore
  UNABLE_TO_CONFIGURE_SENSOR = DaqDeviceExceptionType.unableToConfigureSensor # type: ignore
  UNABLE_TO_GET_SENSOR_CONFIGURATION = DaqDeviceExceptionType.unableToGetSensorConfiguration # type: ignore
  UNABLE_TO_TURN_INTERNAL_TRIGGER_OFF = DaqDeviceExceptionType.unableToTurnInternalTrigger_OFF # type: ignore
  UNABLE_TO_TURN_INTERNAL_TRIGGER_ON = DaqDeviceExceptionType.unableToTurnInternalTrigger_ON # type: ignore
  UNABLE_TO_ENABLE_SENSOR = DaqDeviceExceptionType.unableToEnableSensor # type: ignore
  UNABLE_TO_DISABLE_SENSOR = DaqDeviceExceptionType.unableToDisableSensor # type: ignore
  UNABLE_TO_ENABLE_FOOT_SW_SENSOR = DaqDeviceExceptionType.unableToEnableFootSwSensor # type: ignore
  UNABLE_TO_DISABLE_FOOT_SW_SENSOR = DaqDeviceExceptionType.unableToDisableFootSwSensor # type: ignore
  UNABLE_TO_DETECT_ACCELEROMETER_OFFSET = DaqDeviceExceptionType.unableToDetectAccelerometerOffset # type: ignore
  UNABLE_TO_CHECK_ELECTRODE_IMPEDANCE = DaqDeviceExceptionType.unableToCheckElectrodeImpedance # type: ignore
  UNABLE_TO_GET_ELECTRODE_IMPEDANCE_REPORT = DaqDeviceExceptionType.unableToGetElectrodeImpedanceReport # type: ignore
  UNABLE_TO_TURN_SENSOR_LED_ON = DaqDeviceExceptionType.unableToTurnSensorLedOn # type: ignore
  UNABLE_TO_TURN_FOOTSWSENSOR_LED_ON = DaqDeviceExceptionType.unableToTurnFootSwSensorLedOn # type: ignore
  UNABLE_TO_TURN_ALL_SENSOR_LEDS_ON = DaqDeviceExceptionType.unableToTurnAllSensorLedsOn # type: ignore
  UNABLE_TO_TURN_ALL_SENSOR_LEDS_OFF = DaqDeviceExceptionType.unableToTurnAllSensorLedsOff # type: ignore
  UNABLE_TO_START_SENSOR_MEMORY_RECORDING = DaqDeviceExceptionType.unableToStartSensorMemoryRecording # type: ignore
  UNABLE_TO_STOP_SENSOR_MEMORY_RECORDING = DaqDeviceExceptionType.unableToStopSensorMemoryRecording # type: ignore
  UNABLE_TO_CLEAR_SENSOR_MEMORY = DaqDeviceExceptionType.unableToClearSensorMemory # type: ignore
  UNABLE_TO_FORMAT_SENSOR_MEMORY = DaqDeviceExceptionType.unableToFormatSensorMemory # type: ignore
  UNABLE_TO_START_SENSOR_MEMORY_READING = DaqDeviceExceptionType.unableToStartSensorMemoryReading # type: ignore
  UNABLE_TO_START_SENSOR_SELECTIVE_MEMORY_READING = DaqDeviceExceptionType.unableToStartSensorSelectiveMemoryReading # type: ignore
  UNABLE_TO_STOP_SENSOR_MEMORY_READING = DaqDeviceExceptionType.unableToStopSensorMemoryReading # type: ignore
  UNABLE_TO_ENABLE_SENSOR_MEMORY_MODE = DaqDeviceExceptionType.unableToEnableSensorMemoryMode # type: ignore
  UNABLE_TO_DISABLE_SENSOR_MEMORY_MODE = DaqDeviceExceptionType.unableToDisableSensorMemoryMode # type: ignore
  UNABLE_TO_CALIBRATE_SENSOR_IMU = DaqDeviceExceptionType.unableToCalibrateSensorImu # type: ignore
  UNABLE_TO_CALIBRATE_SENSOR_GYROSCOPE = DaqDeviceExceptionType.unableToCalibrateSensorGyroscope # type: ignore
  UNABLE_TO_SAVE_MP_STATUS = DaqDeviceExceptionType.unableToSaveMPStatus # type: ignore
  UNABLE_TO_GET_FIRMWARE_VERSION = DaqDeviceExceptionType.unableToGetFirmwareVersion # type: ignore
  UNABLE_TO_GET_HARDWARE_VERSION = DaqDeviceExceptionType.unableToGetHardwareVersion # type: ignore
  UNABLE_TO_SET_AUDIO_CONFIGURATION = DaqDeviceExceptionType.unableToSetAudioConfiguration # type: ignore
  UNABLE_TO_GET_AUDIO_CONFIGURATION = DaqDeviceExceptionType.unableToGetAudioConfiguration # type: ignore
  UNABLE_TO_CONVERT_PARAMETER = DaqDeviceExceptionType.unableToConvertParameter # type: ignore
  UNABLE_TO_UPDATE_FIRMWARE = DaqDeviceExceptionType.unableToUpdateFirmware # type: ignore
  UNABLE_TO_GET_FPGA_CONFIG_FLAG = DaqDeviceExceptionType.unableToGetFPGAConfigFlag # type: ignore
  UNABLE_TO_UPDATE_DEVICE_BOARD_EEPROM_INFO = DaqDeviceExceptionType.unableToUpdateDeviceBoardEEPROMInfo # type: ignore
  UNABLE_TO_GET_DEVICE_BOARD_EEPROM_INFO = DaqDeviceExceptionType.unableToGetDeviceBoardEEPROMInfo # type: ignore
  UNABLE_TO_SYNCHRONIZE_DATA = DaqDeviceExceptionType.unableToSynchronizeData # type: ignore
  UNABLE_TO_CHANGE_DEVICE_RF_CHANNEL = DaqDeviceExceptionType.unableToChangeDeviceRFChannel # type: ignore
  UNABLE_TO_CHANGE_SENSOR_RF_CHANNEL = DaqDeviceExceptionType.unableToChangeSensorRFChannel # type: ignore
  UNABLE_TO_GET_DEVICE_RF_CHANNEL = DaqDeviceExceptionType.unableToGetDeviceRFChannel # type: ignore
  UNABLE_TO_SET_FIRST_IMU_CALIBRATION_STEP = DaqDeviceExceptionType.unableToSetFirstImuCalibrationStep # type: ignore
  UNABLE_TO_SET_NEXT_IMU_CALIBRATION_STEP = DaqDeviceExceptionType.unableToSetNextImuCalibrationStep # type: ignore
  UNABLE_TO_SET_IMU_ACQ_TYPE = DaqDeviceExceptionType.unableToSetIMUAcqType # type: ignore
  UNABLE_TO_GET_DEVICE_DEPENDENT_FUNCT_AVAILABILITY = DaqDeviceExceptionType.unableToGetDeviceDependentFunctAvailability # type: ignore
  UNABLE_TO_READ_SENSOR_INFO = DaqDeviceExceptionType.unableToReadSensorInfo # type: ignore


class RFChannelEnum:
  RF_CHANNEL_0 = RFChannel.RFChannel_0 # type: ignore
  RF_CHANNEL_1 = RFChannel.RFChannel_1 # type: ignore
  RF_CHANNEL_2 = RFChannel.RFChannel_2 # type: ignore
  RF_CHANNEL_3 = RFChannel.RFChannel_3 # type: ignore
  RF_CHANNEL_4 = RFChannel.RFChannel_4 # type: ignore
  RF_CHANNEL_5 = RFChannel.RFChannel_5 # type: ignore
  RF_CHANNEL_6 = RFChannel.RFChannel_6 # type: ignore
  RF_CHANNEL_7 = RFChannel.RFChannel_7 # type: ignore


class SamplingRateEnum:
  HZ_2000 = SamplingRate.Hz_2000 # type: ignore # 2 kHz


# NOTE: Raw Gyro and Mag only available during the `RAW_DATA` scheme.
class ImuAcqTypeEnum:
  RAW_DATA = ImuAcqType.RawData # type: ignore 	# Raw IMU data at 284 Hz
  FUSED_9DOF_142HZ = ImuAcqType.Fused9xData_142Hz # type: ignore 	# Quaternion from 9-DOF at 142 Hz 
  FUSED_6DOF_284HZ = ImuAcqType.Fused6xData_284Hz # type: ignore 	# Quaternion from 6-DOF at 284 Hz
  FUSED_9DOF_71HZ = ImuAcqType.Fused9xData_71Hz # type: ignore 	# Quaternion from 9-DOF at 71 Hz
  FUSED_6DOF_142HZ = ImuAcqType.Fused6xData_142Hz # type: ignore 	# Quaternion from 6-DOF at 142 Hz
  MIXED_6DOF_142HZ = ImuAcqType.Mixed6xData_142Hz # type: ignore 	# Quaternion from 6 DOF at 142 Hz and raw acc/gyr at 142 Hz + mag at 47 Hz 


class DataAvailableEventPeriodEnum:
  MS_100 = DataAvailableEventPeriod.ms_100 # type: ignore
  MS_50 = DataAvailableEventPeriod.ms_50 # type: ignore
  MS_25 = DataAvailableEventPeriod.ms_25 # type: ignore
  MS_10 = DataAvailableEventPeriod.ms_10 # type: ignore


class SensorTypeEnum:
  EMG_SENSOR = SensorType.EMG_SENSOR # type: ignore
  INERTIAL_SENSOR = SensorType.INERTIAL_SENSOR # type: ignore
  ANALOG_GP_SENSOR = SensorType.ANALOG_GP_SENSOR # type: ignore
  FSW_SENSOR = SensorType.FSW_SENSOR # type: ignore


class SensorStateEnum:
  BAT_0 = 0x0000
  BAT_33 = 0x0001
  BAT_66 = 0x0002
  BAT_100 = 0x0003


# Gravitational field.
class AccelerometerFullScaleEnum:
  G_2 = AccelerometerFullScale.g_2 # type: ignore
  G_4 = AccelerometerFullScale.g_4 # type: ignore
  G_8 = AccelerometerFullScale.g_8 # type: ignore
  G_16 = AccelerometerFullScale.g_16 # type: ignore


# Degree per second.
class GyroscopeFullScaleEnum:
  DPS_250 = GyroscopeFullScale.dps_250 # type: ignore
  DPS_500 = GyroscopeFullScale.dps_500 # type: ignore
  DPS_1000 = GyroscopeFullScale.dps_1000 # type: ignore
  DPS_2000 = GyroscopeFullScale.dps_2000 # type: ignore


class FootSwProtocolEnum:
  FULL_FOOT = FootSwProtocol.FullFoot # type: ignore
  HALF_FOOT = FootSwProtocol.HalfFoot # type: ignore
  QUARTER_FOOT = FootSwProtocol.QuarterFoot # type: ignore


class SensorCheckReportEnum:
  FAILED = SensorCheckReport.Failed # type: ignore
  PASSED = SensorCheckReport.Passed # type: ignore
  NOT_EXECUTED = SensorCheckReport.NotExecuted # type: ignore
