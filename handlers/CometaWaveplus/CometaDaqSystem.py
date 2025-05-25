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

import sys
from typing import Any, Callable, Iterable
import clr

from handlers.CometaWaveplus.CometaConstants import DataAvailableEventPeriodEnum, RFChannelEnum
from handlers.CometaWaveplus.CometaConstants import DeviceErrorEnum, DeviceStateEnum
from handlers.CometaWaveplus.CometaConstants import SensorCheckReportEnum, SensorTypeEnum
from handlers.CometaWaveplus.CometaCaptureConfiguration import CometaCaptureConfiguration
from handlers.CometaWaveplus.CometaDeviceDependentFunctionalities import CometaDeviceDependentFunctionalities
from handlers.CometaWaveplus.CometaEventArgs import CometaDataAvailableEventArgs, CometaDeviceStateChangedEventArgs, CometaSensorMemoryDataAvailableEventArgs
from handlers.CometaWaveplus.CometaSensorConfiguration import CometaSensorConfiguration
from handlers.CometaWaveplus.CometaVersion import CometaExtVersion, CometaVersion

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from CyUSB import * # type: ignore


class CometaDaqSystem(DaqSystem): # type: ignore
  def get_state(self) -> DeviceStateEnum:
    return self.get_State()

  def get_initial_error(self) -> DeviceErrorEnum:
    return self.get_InitialError()
  
  def get_type(self) -> Iterable[SensorTypeEnum]:
    return self.get_Type()

  def set_capture_configuration(self, capture_config: CometaCaptureConfiguration) -> None:
    self.ConfigureCapture(capture_config)

  def get_capture_configuration(self) -> CometaCaptureConfiguration:
    return self.CaptureConfiguration()

  def start_capturing(self, event_period: DataAvailableEventPeriodEnum) -> None:
    self.StartCapturing(event_period)

  def stop_capturing(self) -> None:
    self.StopCapturing()

  def generate_start_trigger(self) -> None:
    self.GenerateInternalStartTrigger()

  def generate_stop_trigger(self) -> None:
    self.GenerateInternalStopTrigger()

  def get_firmware_version(self) -> Iterable[CometaVersion]:
    return self.get_FirmwareVersion()

  def get_hardware_version(self) -> Iterable[CometaVersion]:
    return self.get_HardwareVersion()

  def get_software_version(self) -> CometaExtVersion:
    return self.get_SoftwareVersion()

  def get_num_installed_sensors(self) -> int:
    return self.get_InstalledSensors()

  def get_num_installed_fsw_sensors(self) -> int:
    return self.get_InstalledFootSwSensors()

  def enable_sensor(self, sensor_id: int) -> None:
    self.EnableSensor(sensor_id)

  def disable_sensor(self, sensor_id: int) -> None:
    self.DisableSensor(sensor_id)

  def enable_fsw_sensors(self) -> None:
    self.EnableFootSwSensors()

  def disable_fsw_sensors(self) -> None:
    self.DisableFootSwSensors()

  def set_sensor_configuration(self, sensor_config: CometaSensorConfiguration, sensor_id: int) -> None:
    self.ConfigureSensor(sensor_config, sensor_id)

  def get_sensor_configuration(self, sensor_id: int) -> CometaSensorConfiguration:
    return self.SensorConfiguration(sensor_id)

  def detect_accelerometer_offset(self, sensor_id: int) -> None:
    self.DetectAccelerometerOffset(sensor_id)

  def check_impedance(self, sensor_id: int) -> Iterable[SensorCheckReportEnum]:
    return self.CheckElectrodeImpedance(sensor_id)

  def turn_led_on(self, sensor_id: int) -> None:
    self.TurnSensorLedOn(sensor_id)

  def turn_all_leds_on(self) -> None:
    self.TurnAllSensorLedsOn()

  def turn_all_leds_off(self) -> None:
    self.TurnAllSensorLedsOff()

  def get_device_dependent_functionalities(self) -> Iterable[CometaDeviceDependentFunctionalities]:
    return self.get_DeviceDependentFunctionalities()

  def add_on_state_changed_handler(self, callback: Callable[[Any, CometaDeviceStateChangedEventArgs], None]) -> None:
    self.StateChanged += callback

  def remove_on_state_changed_handler(self, callback: Callable[[Any, CometaDeviceStateChangedEventArgs], None]) -> None:
    self.remove_StateChanged(callback)

  def add_on_data_available_handler(self, callback: Callable[[Any, CometaDataAvailableEventArgs], None]) -> None:
    self.DataAvailable += callback

  def remove_on_data_available_handler(self, callback: Callable[[Any, CometaDataAvailableEventArgs], None]) -> None:
    self.remove_DataAvailable(callback)

  def add_on_sensor_memory_data_available_handler(self, callback: Callable[[Any, CometaSensorMemoryDataAvailableEventArgs], None]) -> None:
    self.SensorMemoryDataAvailable += callback

  def remove_on_sensor_memory_data_available_handler(self, callback: Callable[[Any, CometaSensorMemoryDataAvailableEventArgs], None]) -> None:
    self.remove_SensorMemoryDataAvailable(callback)

  def start_selective_memory_reading(self, trial_id: int) -> None:
    self.StartSensorSelectiveMemoryReading(trial_id)

  def stop_selective_memory_reading(self) -> None:
    self.StopSensorMemoryReading()

  def dispose(self) -> None:
    self.Dispose()

  def get_master_device_rf_channel(self, device_id: int) -> RFChannelEnum:
    return self.DeviceRFChannel(device_id)

  def set_master_device_rf_channel(self, channel: RFChannelEnum, device_id: int) -> None:
    self.ChangeDeviceRFChannel(channel, device_id)

  def set_semsor_rf_channel(self, channel: RFChannelEnum, device_id) -> None:
    self.ChangeSensorsRFChannel(channel, device_id)

  def write_sync_data(self, data: float, absolute_value: bool) -> None:
    self.WriteSyncData(data, absolute_value)
