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

from nodes.producers.Producer import Producer
from streams import DotsStream

from handlers.MovellaHandler import MovellaFacade
from utils.zmq_utils import *

import numpy as np
from utils.time_utils import get_time
from collections import OrderedDict


######################################
######################################
# A class for streaming Dots IMU data.
######################################
######################################
class DotsStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'dots'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               device_mapping: dict[str, str],
               master_device: str,
               sampling_rate_hz: int = 60,
               num_joints: int = 5,
               is_sync_devices: bool = True,
               is_get_orientation: bool = True,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = None,
               print_status: bool = True,
               print_debug: bool = False,
               timesteps_before_solidified: int = 0,
               **_):

    # Initialize any state that the sensor needs.
    self._sampling_rate_hz = sampling_rate_hz
    self._num_joints = num_joints
    self._master_device = master_device
    self._is_get_orientation = is_get_orientation
    self._is_sync_devices = is_sync_devices
    self._device_mapping = device_mapping
    self._row_id_mapping = OrderedDict([(device_id, row_id) for row_id, device_id in enumerate(self._device_mapping.values())])

    stream_info = {
      "num_joints": self._num_joints,
      "sampling_rate_hz": self._sampling_rate_hz,
      "device_mapping": device_mapping,
      "is_get_orientation": is_get_orientation,
      "timesteps_before_solidified": timesteps_before_solidified
    }

    # Abstract class will call concrete implementation's creation methods
    #   to build the data structure of the sensor
    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     transmit_delay_sample_period_s=transmit_delay_sample_period_s,
                     print_status=print_status, 
                     print_debug=print_debug)


  def create_stream(cls, stream_info: dict) -> DotsStream:
    return DotsStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    self._handler = MovellaFacade(device_mapping=self._device_mapping, 
                                  master_device=self._master_device,
                                  sampling_rate_hz=self._sampling_rate_hz,
                                  is_get_orientation=self._is_get_orientation,
                                  is_sync_devices=self._is_sync_devices)
    # Keep reconnecting until success
    while not self._handler.initialize(): 
      self._handler.cleanup()
    return True


  def _keep_samples(self) -> None:
    self._handler.keep_data()


  def _process_data(self) -> None:
    # Retrieve the oldest enqueued packet for each sensor.
    snapshot = self._handler.get_snapshot()
    if snapshot is not None:
      process_time_s: float = get_time()
      acceleration = np.empty((self._num_joints, 3), dtype=np.float32)
      acceleration.fill(np.nan)
      gyroscope = np.empty((self._num_joints, 3), dtype=np.float32)
      gyroscope.fill(np.nan)
      magnetometer = np.empty((self._num_joints, 3), dtype=np.float32)
      magnetometer.fill(np.nan)
      if self._is_get_orientation:
        quaternion = np.empty((self._num_joints, 4), dtype=np.float32)
        quaternion.fill(np.nan)
      timestamp = np.zeros((self._num_joints), np.uint32)
      toa_s = np.empty((self._num_joints), dtype=np.float64)
      toa_s.fill(np.nan)
      counter = np.zeros((self._num_joints), np.uint32)

      for device, packet in snapshot.items():
        id = self._row_id_mapping[device]
        if packet and packet["acc"].size:
          acceleration[id] = packet["acc"]
          gyroscope[id] = packet["gyr"]
          magnetometer[id] = packet["mag"]
          if self._is_get_orientation:
            quaternion[id] = packet["quaternion"]
          timestamp[id] = packet["timestamp"]
          toa_s[id] = packet["toa_s"]
          counter[id] = packet["counter"]

      data = {
        'acceleration-x': acceleration[:,0],
        'acceleration-y': acceleration[:,1],
        'acceleration-z': acceleration[:,2],
        'gyroscope-x': gyroscope[:,0],
        'gyroscope-y': gyroscope[:,1],
        'gyroscope-z': gyroscope[:,2],
        'magnetometer-x': magnetometer[:,0],
        'magnetometer-y': magnetometer[:,1],
        'magnetometer-z': magnetometer[:,2],
        'timestamp': timestamp,
        'toa_s': toa_s,
        'counter': counter,
      }
      if self._is_get_orientation:
        data["quaternion"] = quaternion

      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data={'dots-imu': data})
    elif not self._is_continue_capture:
      # If triggered to stop and no more available data, send empty 'END' packet and join.
      self._send_end_packet()


  def _stop_new_data(self):
    self._handler.cleanup()


  def _cleanup(self) -> None:
    self._handler.close()
    super()._cleanup()
