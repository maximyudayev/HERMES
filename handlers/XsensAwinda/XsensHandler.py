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
import xsensdeviceapi as xda

from utils.datastructures import NonOverflowingCounterAlignedFifoBuffer
from utils.time_utils import get_time


class AwindaDataCallback(xda.XsCallback): # type: ignore
  def __init__(self,
               on_each_packet_received: Callable[[float, Any, Any], None]):
    super().__init__()
    self._on_each_packet_received = on_each_packet_received


  # How are interpolated packets for previous time steps provided?
  def onLiveDataAvailable(self, device, packet):
    self._on_each_packet_received(get_time(), device, packet)


class AwindaConnectivityCallback(xda.XsCallback): # type: ignore
  def __init__(self,
               on_wireless_device_connected: Callable[[Any], None]):
    super().__init__()
    self._on_wireless_device_connected = on_wireless_device_connected


  def onConnectivityChanged(self, dev, newState):
    # TODO: add additional logic in case devices disconnect, etc.
    if newState == xda.XCS_Wireless: # type: ignore
      self._on_wireless_device_connected(dev)


class XsensFacade:
  def __init__(self,
               device_mapping: dict[str, str],
               radio_channel: int,
               sampling_rate_hz: int,
               timesteps_before_stale: int = 100) -> None:
    # Queue used to synchronize current main thread and callback handler thread listening 
    #   to device connections when all expected devices connected before continuing
    self._is_all_connected_queue = queue.Queue(maxsize=1)
    self._device_connection_status = dict.fromkeys(list(device_mapping.values()), False)
    self._radio_channel = radio_channel
    self._sampling_rate_hz = sampling_rate_hz
    self._is_keep_data = False
    self._buffer = NonOverflowingCounterAlignedFifoBuffer(keys=device_mapping.values(),
                                                          timesteps_before_stale=timesteps_before_stale,
                                                          num_bits_timestamp=16)
    self._packet_queue = queue.Queue()
    self._is_more = True


  def initialize(self) -> bool:
    self._is_measuring = True
    self._control = xda.XsControl.construct() # type: ignore
    port_info_array = xda.XsScanner.scanPorts() # type: ignore

    # Open the detected devices and pick the Awinda station
    try:
      master_port_info = next((port for port in port_info_array if port.deviceId().isWirelessMaster()))
    except Exception as _:
      return False

    if not self._control.openPort(master_port_info.portName(), master_port_info.baudrate()):
      return False
    
    # Get the device object
    self._master_device = self._control.device(master_port_info.deviceId())
    if not self._master_device:
      return False

    def on_wireless_device_connected(dev) -> None:
      device_id: str = str(dev.deviceId())
      self._device_connection_status[device_id] = True
      print("Connected to %s"%device_id, flush=True)
      if all(self._device_connection_status.values()): self._is_all_connected_queue.put(True)

    def on_each_packet_received(toa_s, device, packet) -> None:
      if self._is_keep_data:
        device_id: str = str(device.deviceId())
        acc = packet.calibratedAcceleration()
        gyr = packet.calibratedGyroscopeData()
        mag = packet.calibratedMagneticField()
        quaternion = packet.orientationQuaternion()
        timestamp = packet.sampleTimeFine()
        counter = packet.packetCounter()
        data = {
          "device_id":            device_id,
          "acc":                  acc,
          "gyr":                  gyr,
          "mag":                  mag,
          "quaternion":           quaternion,
          "toa_s":                toa_s,
          "timestamp":            timestamp,
          "counter_onboard":      counter,
        }
        self._packet_queue.put({"key": device_id, "data": data, "counter": counter})

    # Register event handler on the main device
    self._conn_callback = AwindaConnectivityCallback(on_wireless_device_connected=on_wireless_device_connected)
    self._master_device.addCallbackHandler(self._conn_callback)

    # Enable radio to accept connections from the sensors
    if self._master_device.isRadioEnabled():
      if not self._master_device.disableRadio():
        return False
    if not self._master_device.enableRadio(self._radio_channel):
      return False

    # Will block the current thread until the Awinda onConnectivityChanged has changed to XCS_Wireless for all expected devices
    self._is_all_connected_queue.get()

    # Put devices in Config Mode and request desired data and rate
    self._master_device.gotoConfig()
    config_array = xda.XsOutputConfigurationArray() # type: ignore
    # For data that accompanies every packet (timestamp, status, etc.), the selected sample rate will be ignored
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_PacketCounter, self._sampling_rate_hz)) # type: ignore
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_SampleTimeFine, self._sampling_rate_hz)) # type: ignore
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, self._sampling_rate_hz)) # type: ignore
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, self._sampling_rate_hz)) # type: ignore
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, self._sampling_rate_hz)) # type: ignore # NOTE: also has XDI_MagneticFieldCorrected
    config_array.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, self._sampling_rate_hz)) # type: ignore
    
    if not self._master_device.setOutputConfiguration(config_array):
      print("Could not configure the Awinda master device. Aborting.", flush=True)
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
            print("No more packets from Xsens SDK, flush buffers into the output Queue.", flush=True)
            self._buffer.flush()
            break

    self._packet_funneling_thread = threading.Thread(target=funnel_packets, args=(self._packet_queue,))

    # Register listener of new data
    self._data_callback = AwindaDataCallback(on_each_packet_received=on_each_packet_received)
    self._master_device.addCallbackHandler(self._data_callback)

    # Put all devices connected to the Awinda station into Measurement Mode
    # NOTE: Will begin trigerring the callback and saving data, while awaiting the SYNC signal from the Broker
    if not self._master_device.gotoMeasurement():
      print("Could not set Awinda master to measurement mode. Aborting.", flush=True)
      return False

    self._packet_funneling_thread.start()
    return True


  def keep_data(self) -> None:
    self._is_keep_data = True


  def get_snapshot(self) -> dict[str, dict | None] | None:
    return self._buffer.yeet()


  def cleanup(self) -> None:
    self._control.close()
    self._is_more = False
    self._control.destruct()

  
  def close(self) -> None:
    self._packet_funneling_thread.join()
