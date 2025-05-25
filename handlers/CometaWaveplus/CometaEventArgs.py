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
from typing import Iterable
import clr

from CometaConstants import DeviceErrorEnum, DeviceStateEnum, SensorStateEnum

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from CyUSB import * # type: ignore


class CometaCommandProgressEventArgs(CommandProgressEventArgs): # type: ignore
  def get_progress(self) -> int:
    return self.ProgressInPercent 


class CometaDeviceStateChangedEventArgs(DeviceStateChangedEventArgs): # type: ignore
  def get_state(self) -> DeviceStateEnum:
    return self.State


class CometaDataAvailableEventArgs(DataAvailableEventArgs): # type: ignore
  def scan_number(self) -> int:
    return self.ScanNumber
  
  def get_emg_samples(self) -> Iterable[Iterable[float]]:
    return self.Samples

  def get_orientation_samples(self) -> Iterable[Iterable[tuple[float, float, float, float]]]:
    # NOTE: Only available for ImuAcqTypeEnum.FUSED_... and ImuAcqTypeEnum.MIXED_6DOF_142HZ.
    return self.ImuSamples

  def get_accelerometer_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    # NOTE: Only available for ImuAcqTypeEnum.RAW_DATA and ImuAcqTypeEnum.MIXED_6DOF_142HZ.
    return self.AccelerometerSamples

  def get_gyroscope_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    # NOTE: Only available for ImuAcqTypeEnum.RAW_DATA and ImuAcqTypeEnum.MIXED_6DOF_142HZ.
    return self.GyroscopeSamples

  def get_magnetometer_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    # NOTE: Only available for ImuAcqTypeEnum.RAW_DATA and ImuAcqTypeEnum.MIXED_6DOF_142HZ.
    return self.MagnetometerSamples

  def get_sync_samples(self):
    return self.SyncSamples

  def get_sensor_states(self) -> Iterable[Iterable[SensorStateEnum]]:
    # NOTE: NOT available for ImuAcqTypeEnum.FUSED_....
    return self.SensorStates

  def get_fsw_samples(self) -> Iterable[tuple[int, int]]:
    return self.FootSwSamples

  def get_fsw_raw_samples(self) -> Iterable[tuple[int, int]]:
    return self.FootSwRawSamples

  def get_fsw_sensor_states(self) -> Iterable[Iterable[SensorStateEnum]]:
    return self.FootSwSensorStates

  def is_start_trigger_detected(self) -> bool:
    # NOTE: True if start trigger was detected.
    return self.StartTriggerDetected

  def is_stop_trigger_detected(self) -> bool:
    # NOTE: True if stop trigger was detected.
    return self.StopTriggerDetected

  def start_trigger_scan(self) -> int:
    # NOTE: Number of the scansion corresponding to the start trigger detection.
    return self.StartTriggerScan

  def stop_trigger_scan(self) -> int:
    # NOTE: Number of the scansion corresponding to the stop trigger detection.
    return self.StopTriggerScan

  def get_transfer_rate(self) -> int:
    return self.DataTransferRate

  def get_sensor_rf_lost_packets(self) -> Iterable[int]:
    return self.SensorRFLostPackets

  def get_usb_lost_packets(self) -> int:
    return self.USBLostPackets


class CometaSensorMemoryDataAvailableEventArgs(SensorMemoryDataAvailableEventArgs): # type: ignore
  def get_num_samples(self) -> int:
    return self.SamplesNumber

  def get_emg_samples(self) -> Iterable[Iterable[float]]:
    return self.Samples

  def get_orientation_samples(self) -> Iterable[Iterable[tuple[float, float, float, float]]]:
    return self.ImuSamples

  def get_accelerometer_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    return self.AccelerometerSamples

  def get_gyroscope_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    return self.GyroscopeSamples

  def get_magnetometer_samples(self) -> Iterable[Iterable[tuple[float, float, float]]]:
    return self.MagnetometerSamples

  def get_sensor_states(self) -> Iterable[Iterable[SensorStateEnum]]:
    return self.SensorStates

  def get_fsw_samples(self) -> Iterable[tuple[int, int]]:
    return self.FootSwSamples

  def get_fsw_raw_samples(self) -> Iterable[tuple[int, int]]:
    return self.FootSwRawSamples

  def get_fsw_sensor_states(self) -> Iterable[Iterable[SensorStateEnum]]:
    return self.FootSwSensorStates

  def is_trial_end(self) -> bool:
    return self.TrialEnd
  
  def get_num_saved_trials(self) -> int:
    return self.SavedTrialsNumber
  
  def get_transfer_progress(self) -> int:
    return self.TransferredSamplesInPercent

  def get_current_trial_transfer_progress(self) -> int:
    return self.CurrentTrialTransferredSamplesInPercent

  def get_current_trial_id(self) -> int:
    return self.CurrentTrial
  
  def get_transfer_rate(self) -> int:
    return self.DataTransferRate

  def get_sensor_lost_packets(self) -> Iterable[int]:
    return self.SensorLostPackets
  
  def get_error_code(self) -> DeviceErrorEnum:
    return self.ErrorCode
  
  def get_lost_packets(self) -> int:
    return self.LostPackets
