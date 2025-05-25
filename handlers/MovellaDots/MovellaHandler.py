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
import threading
from typing import Any, Callable
import movelladot_pc_sdk as mdda
from collections import OrderedDict

from handlers.MovellaDots.MovellaConstants import MOVELLA_LOGGING_MODE, MOVELLA_PAYLOAD_MODE
from utils.datastructures import TimestampAlignedFifoBuffer
from utils.user_settings import *
from utils.time_utils import get_time


class DotDataCallback(mdda.XsDotCallback): # type: ignore
  def __init__(self,
               on_packet_received: Callable[[float, Any, Any], None]):
    super().__init__()
    self._on_packet_received = on_packet_received


  def onLiveDataAvailable(self, device, packet):
    self._on_packet_received(get_time(), device, packet)


class DotConnectivityCallback(mdda.XsDotCallback): # type: ignore
  def __init__(self,
               on_advertisement_found: Callable,
               on_device_disconnected: Callable):
    super().__init__()
    self._on_advertisement_found = on_advertisement_found
    self._on_device_disconnected = on_device_disconnected


  def onAdvertisementFound(self, port_info):
    self._on_advertisement_found(port_info)


  def onDeviceStateChanged(self, device, new_state, old_state):
    if new_state == mdda.XDS_Destructing: # type: ignore
      self._on_device_disconnected(device)


  def onError(self, result, error):
    print(error)


class MovellaFacade:
  def __init__(self,
               device_mapping: dict[str, str],
               master_device: str,
               sampling_rate_hz: int,
               payload_mode: str = 'RateQuantitieswMag',
               logging_mode: str = 'Euler',
               filter_profile: str = 'General',
               is_sync_devices: bool = True,
               is_enable_logging: bool = False,
               timesteps_before_stale: int = 100) -> None:
    self._is_all_discovered_queue = queue.Queue(maxsize=1)
    self._device_mapping = device_mapping
    self._discovered_devices = list()
    self._connected_devices: OrderedDict[str, Any] = OrderedDict([(v, None) for v in device_mapping.values()])
    sampling_period = round(1/sampling_rate_hz * 10000)
    self._buffer = TimestampAlignedFifoBuffer(keys=device_mapping.values(),
                                              timesteps_before_stale=timesteps_before_stale,
                                              sampling_period=sampling_period,
                                              num_bits_timestamp=32)
    self._packet_queue = queue.Queue()
    self._is_more = True
    self._master_device_id = device_mapping[master_device]
    self._sampling_rate_hz = sampling_rate_hz
    self._is_sync_devices = is_sync_devices
    self._is_enable_logging = is_enable_logging
    self._is_keep_data = False
    self._filter_profile = filter_profile
    self._payload_mode = MOVELLA_PAYLOAD_MODE[payload_mode]
    self._logging_mode = MOVELLA_LOGGING_MODE[logging_mode]


  def initialize(self) -> bool:
    # Create connection manager
    self._manager = mdda.XsDotConnectionManager() # type: ignore
    if self._manager is None:
      return False

    def on_advertisement_found(port_info) -> None:
      if not port_info.isBluetooth(): return
      self._discovered_devices.append(port_info)
      if len(self._discovered_devices) == len(self._device_mapping): self._is_all_discovered_queue.put(True)
      print("discovered %s"%port_info.bluetoothAddress(), flush=True)

    def on_packet_received(toa_s, device, packet):
      if self._is_keep_data:
        device_id: str = str(device.deviceId())
        timestamp = packet.sampleTimeFine()
        data = {"device_id": device_id,
                "timestamp": timestamp,
                "toa_s": toa_s}
        for data_name, data_getter in self._payload_mode["methods"].items():
          data[data_name] = data_getter["func"](packet)
        self._packet_queue.put({"key": device_id, "data": data, "timestamp": timestamp})

    def on_device_disconnected(device):
      device_id: str = str(device.deviceId())
      print("%s disconnected"%device_id, flush=True)
      self._connected_devices[device_id] = None

    # Attach callback handler to connection manager
    self._conn_callback = DotConnectivityCallback(on_advertisement_found=on_advertisement_found,
                                                  on_device_disconnected=on_device_disconnected)
    self._manager.addXsDotCallbackHandler(self._conn_callback)

    # Start a scan and wait until we have found all devices
    self._manager.enableDeviceDetection()
    self._is_all_discovered_queue.get()
    self._manager.disableDeviceDetection()

    for port_info in self._discovered_devices:
      if not self._manager.openPort(port_info): 
        print("failed to connect to %s"%port_info.bluetoothAddress(), flush=True)
        return False
      device = self._manager.device(port_info.deviceId())
      device_id: str = str(port_info.deviceId())
      self._connected_devices[device_id] = device
      print("connected to %s"%port_info.bluetoothAddress(), flush=True)

    # Make sure all connected devices have the same filter profile and output rate
    for device_id, device in self._connected_devices.items():
      if not device.setOnboardFilterProfile(self._filter_profile):
        return False
      if not device.setOutputRate(self._sampling_rate_hz):
        return False

    # Call facade sync function, not directly the backend manager proxy
    if self._is_sync_devices:
      if not self._sync(attempts=3):
        return False

    if self._is_enable_logging:
      for device_id, device in self._connected_devices.items():
        device.setLogOptions(self._logging_mode)
        logFileName = "logfile_" + device.bluetoothAddress().replace(':', '-') + ".csv"
        print(f"Enable logging to: {logFileName}", flush=True)
        if not device.enableLogging(logFileName):
          print(f"Failed to enable logging. Reason: {device.lastResultText()}", flush=True)
          return False

    # Set dots to streaming mode and break out of the loop if successful.
    if not self._stream():
      return False

    # Funnels packets from the background thread-facing interleaved Queue of async packets, 
    #   into aligned Deque datastructure.
    def funnel_packets(packet_queue: queue.Queue, timeout: float = 5.0):
      while True:
        try:
          next_packet = packet_queue.get(timeout=timeout)
          self._buffer.plop(**next_packet)
        except queue.Empty:
          if self._is_more:
            continue
          else:
            print("No more packets from Movella SDK, flush buffers into the output Queue.", flush=True)
            self._buffer.flush()
            break

    self._packet_funneling_thread = threading.Thread(target=funnel_packets, args=(self._packet_queue,))

    self._data_callback = DotDataCallback(on_packet_received=on_packet_received)
    self._manager.addXsDotCallbackHandler(self._data_callback)

    self._packet_funneling_thread.start()
    return True


  def _sync(self, attempts=1) -> bool:
    # NOTE: Syncing may not work on some devices due to poor BT drivers.
    while attempts > 0:
      print(f"{attempts} attempts left to sync DOTs.", flush=True)
      if self._manager.startSync(self._connected_devices[self._master_device_id].bluetoothAddress()):
        return True
      else:
        attempts -= 1
        self._manager.stopSync()
    return False


  def _stream(self) -> bool:
    # Start live data output. Make sure root node is last to go to measurement.
    # NOTE: orientation reset works only in 'yaw' direction on DOTs -> no reason to use, turn on flat on the table, then attach to body and start program.
    ordered_device_list: list[tuple[str, Any]] = [*[(device_id, device) for device_id, device in self._connected_devices.items()
                                                        if device_id != self._master_device_id], 
                                                    (self._master_device_id, self._connected_devices[self._master_device_id])]
    for (joint, device) in ordered_device_list:
      if not device.startMeasurement(self._payload_mode["payload_mode"]):
        return False
    return True


  def keep_data(self) -> None:
    self._is_keep_data = True


  def get_snapshot(self) -> dict[str, dict | None] | None:
    return self._buffer.yeet()


  def cleanup(self) -> None:
    for device_id, device in self._connected_devices.items():
      if device is not None:
        if not device.stopMeasurement():
          print("Failed to stop measurement.", flush=True)
        if self._is_enable_logging and not device.disableLogging():
          print("Failed to disable logging.", flush=True)
        self._connected_devices[device_id] = None
    self._is_more = False
    self._discovered_devices = list()
    if self._is_sync_devices:
      self._manager.stopSync()


  def close(self) -> None:
    self._manager.close()
    self._packet_funneling_thread.join()
