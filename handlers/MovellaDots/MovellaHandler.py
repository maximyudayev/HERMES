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
from utils.datastructures import TimestampAlignedFifoBuffer #type: ignore
from utils.user_settings import *
from utils.time_utils import get_time

from bleak import BleakScanner, BleakClient # type: ignore
from movella_dot_py.core.sensor import MovellaDOTSensor # type: ignore
from movella_dot_py.models.data_structures import SensorConfiguration # type: ignore
from movella_dot_py.models.enums import OutputRate, FilterProfile, PayloadMode # type: ignore
import asyncio
import time


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
               mac_mapping: dict[str, str],
               master_device: str,
               sampling_rate_hz: int,
               payload_mode: str = 'RateQuantitieswMag',
               logging_mode: str = 'Euler',
               filter_profile: str = 'General',
               is_sync_devices: bool = True,
               is_enable_logging: bool = False,
               timesteps_before_stale: int = 100) -> None:
    self._is_all_discovered_queue = queue.Queue(maxsize=1)
    self._device_mapping = dict(zip(device_mapping.values(), device_mapping.keys()))
    self._mac_mapping = dict(zip(mac_mapping.values(), mac_mapping.keys()))
    self._discovered_devices: OrderedDict[str, Any] = OrderedDict([(v, None) for v in mac_mapping.values()])
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
    self._payload_mode = payload_mode
    _PAYLOAD_MODE_MAP = {"RateQuantitieswMag": "RATE_QUANTITIES_WITH_MAG"}
    self._payload_mode_enum = getattr(PayloadMode, _PAYLOAD_MODE_MAP[payload_mode])
    self._sensors: dict[str, MovellaDOTSensor] = {}   
    self._last_idx: dict[str, int] = {}  
    self._loop = None
    self._loop_thread = None
    self._device_id_by_joint = device_mapping.copy()


  def _ensure_loop(self):
    if getattr(self, "_loop", None) is None or not self._loop.is_running(): #type: ignore
      self._loop = asyncio.new_event_loop()
      self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
      self._loop_thread.start()

  def _run(self, coro):
    fut = asyncio.run_coroutine_threadsafe(coro, self._loop) #type: ignore
    return fut.result()

  def initialize(self) -> bool:
    cfg = SensorConfiguration(
      output_rate=getattr(OutputRate, f"RATE_{self._sampling_rate_hz}"),
      filter_profile=getattr(FilterProfile, self._filter_profile.upper()),
      payload_mode = self._payload_mode_enum)
    self._cfg = cfg
    self._ensure_loop()

    self._ensure_loop()
    device_id_by_joint = self._device_id_by_joint

    devices = self._run(BleakScanner.discover(timeout=5.0))
    for d in devices:
      address = d.address.upper()
      mac_no_colon = ''.join(address.split(':'))
      if mac_no_colon in self._mac_mapping.keys():
        self._discovered_devices[mac_no_colon] = d
        print(f"discovered {self._mac_mapping[mac_no_colon]}", flush=True)
      else:
        print(f"discovered {address}", flush=True)

    # Require that all configured devices are discovered
    if all(self._discovered_devices.values()):
      self._is_all_discovered_queue.put(True)
    else:
      missing = [mac for mac, dev in self._discovered_devices.items() if dev is None]
      print(f"Missing configured devices: {missing}", flush=True)
      return False

    for mac_no_colon, bleak_dev in self._discovered_devices.items():
      joint = self._mac_mapping.get(mac_no_colon) 
      if joint is None:
        # skip not configured devices
        continue

      device_id = device_id_by_joint.get(joint) 
      if device_id is None:
        print(f"no device_id for joint '{joint}'", flush=True)
        continue
      try:
        sensor = MovellaDOTSensor(cfg)
        sensor.client = BleakClient(bleak_dev.address)

        self._run(sensor.client.connect())
        sensor.is_connected = True
        sensor._device_address = bleak_dev.address
        sensor._device_name = bleak_dev.name

        self._run(sensor.configure_sensor())

        self._sensors[device_id] = sensor
        self._connected_devices[device_id] = sensor
        self._last_idx[device_id] = 0

        print(f"connected to {mac_no_colon}: {device_id}", flush=True)
      except Exception as e:
        print(f"failed to connect/configure {bleak_dev.address.upper()}: {e}", flush=True)
        self._connected_devices[device_id] = None

    if not any(self._connected_devices.values()):
      print("No dots successfully connected.", flush=True)
      return False
    
    if not self._stream():
      return False
    def _collector_packets():
      field_map = {
        "accelerations" : "acceleration",
        "angular_velocity" : "gyroscope",
        "magnetic_field" : "magnetometer"}
      
      while True:
        # no packets until keep_data() is called
        if not self._is_keep_data:
          time.sleep(0.02)
          continue

        for device_id, sensor in list(self._sensors.items()):
          if sensor is None:
            continue
          sample = sensor.get_collected_data()
          if not sample:
            continue

          ts = sample.get("timestamps")
          if ts is None or ts.size == 0:
              continue

          start = self._last_idx.get(device_id, 0)
          end = int(ts.shape[0])
          if end <= start:
            continue

          cols = {dst: sample.get(src) for src, dst in field_map.items()}
          for i in range(start, end):
            ts_i = int(ts[i])
            data = {
              "device_id": device_id,
              "timestamp": ts_i,  
              "toa_s": get_time(),   
            }
            for dst_key, col in cols.items():
              if col is not None and i < col.shape[0]:
                data[dst_key] = col[i]
            self._packet_queue.put({"key": device_id, "data": data, "timestamp": ts_i})

          self._last_idx[device_id] = end

          time.sleep(0.02)

    self._collector_thread = threading.Thread(target=_collector_packets, daemon=True)
    self._collector_thread.start()

    def funnel_packets(packet_queue: queue.Queue, timeout: float = 5.0):
      while True:
        try:
          next_packet = packet_queue.get(timeout=timeout)
          self._buffer.plop(**next_packet)
        except queue.Empty:
          continue

    self._packet_funneling_thread = threading.Thread(target=funnel_packets, args=(self._packet_queue,), daemon=True)
    self._packet_funneling_thread.start()

    return True
  
  def _stream(self) -> bool:
    ordered_device_list = [*[(dev_id, s) for dev_id, s in self._connected_devices.items()
                              if dev_id != self._master_device_id],
                           (self._master_device_id, self._connected_devices[self._master_device_id])]
    for (dev_id, sensor) in ordered_device_list:
      if sensor is None:
        continue
      try:
        self._run(sensor.start_measurement())
      except Exception as e:
        print(f"Failed to start measurement for {dev_id}: {e}", flush=True)
        return False
    return True


  def keep_data(self) -> None:
    self._is_keep_data = True
    for dev_id, sensor in self._sensors.items():
      sample = sensor.get_collected_data() or {}
      ts = sample.get("timestamps")
      self._last_idx[dev_id] = int(ts.shape[0]) if ts is not None else 0


  def get_snapshot(self) -> dict[str, dict | None] | None:
    return self._buffer.yeet()


  # TODO: adapt cleanup and close functions to new dots reader
  def cleanup(self) -> None:
    for device_id, device in self._connected_devices.items():
      if device is not None:
        if not device.stopMeasurement():
          print("Failed to stop measurement.", flush=True)
        if self._is_enable_logging and not device.disableLogging():
          print("Failed to disable logging.", flush=True)
        self._connected_devices[device_id] = None
    self._is_more = False
    self._discovered_devices = OrderedDict([(v, None) for v in self._mac_mapping.keys()])
    # if self._is_sync_devices:
    #   self._manager.stopSync()


  def close(self) -> None:
    # self._manager.close()
    self._packet_funneling_thread.join()