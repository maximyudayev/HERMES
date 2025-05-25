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

import queue
import time
from typing import Any, Callable

from handlers.CometaWaveplus.CometaConstants import AccelerometerFullScaleEnum, GyroscopeFullScaleEnum, ImuAcqTypeEnum
from handlers.CometaWaveplus.CometaConstants import SamplingRateEnum, DataAvailableEventPeriodEnum 
from handlers.CometaWaveplus.CometaConstants import DeviceStateEnum
from handlers.CometaWaveplus.CometaConstants import SensorCheckReportEnum, SensorTypeEnum
from handlers.CometaWaveplus.CometaCaptureConfiguration import CometaCaptureConfiguration
from handlers.CometaWaveplus.CometaDaqSystem import CometaDaqSystem
from handlers.CometaWaveplus.CometaEventArgs import CometaDataAvailableEventArgs, CometaDeviceStateChangedEventArgs
from handlers.CometaWaveplus.CometaSensorConfiguration import CometaSensorConfiguration
from utils.time_utils import get_time
import numpy as np

import sys
import clr

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from CyUSB import * # type: ignore

# Waveplus device is EMG+Acc, no IMU data.
class CometaFacade:
  def __init__(self,
              device_mapping: dict[str, str],
              is_check_impedance: bool = False):
    self._device_mapping = device_mapping
    self._sensor_ids = list(map(lambda s: int(s)-1, device_mapping.values()))
    self._is_check_impedance = is_check_impedance
    self._is_more = True
    self._packet_queue = queue.Queue()
    self._on_data_available_fn: Callable[[Any, DeviceStateChangedEventArgs], None] = self._idle_on_data_callback # type: ignore


  def initialize(self) -> bool:
    self._daq = CometaDaqSystem()
    if self._daq.get_state() == DeviceStateEnum.NOT_CONNECTED:
      print('Cometa DAQ is not plugged in or is OFF. Connect it properly.', flush=True)
      return False

    # Configure each sensor as desired, identically.
    # TODO: use user-provided parameters.
    sensor_config = CometaSensorConfiguration()
    sensor_config.set_sensor_type(SensorTypeEnum.EMG_SENSOR)
    sensor_config.set_accelerometer_full_scale(AccelerometerFullScaleEnum.G_8)
    sensor_config.set_gyroscope_full_scale(GyroscopeFullScaleEnum.DPS_1000)

    # Disable all discovered sensors.
    self._daq.disable_sensor(0)
    for sensor_id in self._device_mapping.values():
      # Enable desired sensors and light up LED during setup.
      self._daq.enable_sensor(int(sensor_id))
      self._daq.set_sensor_configuration(sensor_config, int(sensor_id))
      # TODO: Read why/if LED couldn't come on.
      print('EMG #%s connected'%sensor_id, flush=True)

    # DAQ-wide configurations.
    capture_config = CometaCaptureConfiguration()
    capture_config.set_imu_acq_type(ImuAcqTypeEnum.RAW_DATA)
    capture_config.set_sampling_rate(SamplingRateEnum.HZ_2000)
    self._daq.set_capture_configuration(capture_config)

    # Check that the sensors are properly attached.
    if self._is_check_impedance:
      is_impedance_checked = False
      while not is_impedance_checked:
        input('Place the sensors on the body and press any key to check impedance.')
        impedance_results = self._daq.check_impedance(0)
        if all(map(lambda result: result == SensorCheckReportEnum.PASSED, impedance_results)):
          is_impedance_checked = True
        else:
          failed_ids, _ = zip(*filter(lambda tup: tup[1] == SensorCheckReportEnum.FAILED, enumerate(impedance_results)))
          print('EMG sensors %s failed impedance check, retry.'%str(failed_ids), flush=True)

    # Add event handlers to consume data and to capture DAQ FSM state change.
    def on_state_changed(source, args: DeviceStateChangedEventArgs) -> None: # type: ignore
      new_state = args.State
      if new_state == DeviceStateEnum.NOT_CONNECTED: 
        print('DAQ not connected %s.'%new_state, flush=True)
      elif new_state == DeviceStateEnum.INITIALIZING: 
        print('DAQ initializing %s.'%new_state, flush=True)
      elif new_state == DeviceStateEnum.COMMUNICATION_ERROR:
        print('DAQ encountered communication error %s.'%new_state, flush=True)
      elif new_state == DeviceStateEnum.INITIALIZING_ERROR:
        print('DAQ encountered initialization error %s.'%new_state, flush=True)
      elif new_state == DeviceStateEnum.IDLE:
        print('DAQ successfully in idle %s.'%new_state, flush=True)
      elif new_state == DeviceStateEnum.CAPTURING:
        print('DAQ started capturing data %s.'%new_state, flush=True)
      else: 
        print('DAQ in unknown state %s.'%new_state, flush=True)
      time.sleep(1)

    def on_data_available(source, args: DataAvailableEventArgs) -> None: # type: ignore
      self._on_data_available_fn(source, args)

    self._daq.add_on_state_changed_handler(on_state_changed)
    self._daq.add_on_data_available_handler(on_data_available)

    time.sleep(10)
    self._daq.start_capturing(DataAvailableEventPeriodEnum.MS_100)

    return True


  def keep_data(self) -> None:
    self._on_data_available_fn = self._active_on_data_callback


  def _idle_on_data_callback(self, source, args: DataAvailableEventArgs) -> None: # type: ignore
    return


  def _active_on_data_callback(self, source, args: DataAvailableEventArgs) -> None: # type: ignore
    toa_s = get_time()
    # NOTE: SDK duplicates Acc measurements 14x times to match the vector length of the 2kHz EMG.
    num_samples = args.ScanNumber
    emg_samples = args.Samples
    emg = np.array(emg_samples)[self._sensor_ids].transpose((1,0))
    data = {
      'emg': emg,
      'num_samples': num_samples,
      'toa_s': toa_s,
    }
    self._packet_queue.put({'cometa-emg': data})


  def get_packet(self) -> dict | None:
    try:
      return self._packet_queue.get(timeout=5.0)
    except queue.Empty:
      print("No more packets from Cometa SDK.")
      return None


  def cleanup(self) -> None:
    self._daq.stop_capturing()
    self._is_more = False


  def close(self) -> None:
    self._daq.disable_sensor(0)
    # self._daq.remove_on_data_available_handler()
    # self._daq.remove_on_state_changed_handler()
    # self._daq.remove_on_sensor_memory_data_available_handler()
    self._daq.dispose()
