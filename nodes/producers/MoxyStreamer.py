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
from streams import MoxyStream

from openant.easy.node import Node as AntNode
from openant.devices.scanner import Scanner
from openant.devices import ANTPLUS_NETWORK_KEY

import queue
from utils.print_utils import *
from utils.zmq_utils import *
from utils.time_utils import get_time


class MoxyAntNode(AntNode):
  def discover_devices(self, expected_devices: list[str]) -> bool:
    timeout_s = 5
    end_time = get_time() + timeout_s
    self.devices = set()
    while get_time() < end_time:
      try:
        data_type, channel, data = self._datas.get(timeout=1.0)
        self._datas.task_done()
        if data_type == "broadcast":
          byte_data = bytes(data)
          id = str(byte_data[9] + (byte_data[10] << 8))
          if id not in self.devices:
            print(f"device: {id} found", flush=True)
            self.channels[channel].on_broadcast_data(data)
            self.devices.add(id)
      except queue.Empty:
        pass
    
    if len(self.devices) == len(expected_devices):
      return True
    else:
      return False


class MoxyStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'moxy'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               devices: list[str],
               sampling_rate_hz: float = 0.5,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = float('nan'),
               **_):
    self._devices = devices
    self._previous_counters: dict[str, int | None] = {dev: None for dev in devices}

    stream_info = {
      "devices": devices,
      "sampling_rate_hz": sampling_rate_hz
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     sampling_rate_hz=sampling_rate_hz,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     transmit_delay_sample_period_s=transmit_delay_sample_period_s)


  @classmethod
  def create_stream(cls, stream_info: dict) -> MoxyStream:
    return MoxyStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    self.node = MoxyAntNode()
    self.node.set_network_key(0x00, ANTPLUS_NETWORK_KEY)

    # NOTE: the Scanner object is not needed with the custom Node,
    #       unless we want to filter Ant devices (e.g. potentially multiple foreign devices).
    self.scanner = Scanner(self.node, device_id=0, device_type=0)

    def on_update(device_tuple, common):
      device_id = device_tuple[0]
      print(f"Device #{device_id} common data update: {common}", flush=True)

    def on_device_data(device, page_name, data):
      print(f"Device: {device}, broadcast: {page_name}, data: {data}", flush=True)

    def on_found(device_tuple):
      print("Device found", flush=True)

    self.scanner.on_found = on_found
    self.scanner.on_update = on_update
    return self.node.discover_devices(self._devices)


  def _keep_samples(self) -> None:
    # Clear the buffer queue of accumulated values during the system bring-up.
    try:
      while True:
        self.node._datas.get_nowait()
    except queue.Empty:
      return


  def _process_data(self):
    try:
      data_type, channel, data = self.node._datas.get(timeout=5.0)
      process_time_s: float = get_time()
      self.node._datas.task_done()

      # TODO: check the logic here.
      if data_type == "broadcast" and data[0] == 1:
        byte_data = bytes(data)
        device_id = str(byte_data[9] + (byte_data[10] << 8))
        # Don't process the packet if it didn't come from one of the expected devices.
        if device_id in self._devices: return
        
        THb = ((int(data[4] >> 4) << 4) + (int(data[4] % 2**4)) + (int(data[5] % 2**4) << 8)) * 0.01
        SmO2 = ((int(data[7] >> 4) << 6) + (int(data[7] % 2**4) << 2) + int(data[6] % 2**4)) * 0.1
        counter = data[1]

        # TODO: check if this is necessary. Does Ant node put in the queue same packets multiple times?
        if self._previous_counters[device_id] != counter:
          tag: str = "%s.%s.data" % (self._log_source_tag(), device_id)
          data = {
            'THb': THb,
            'SmO2': SmO2,
            'counter': counter,
          }
          self._publish(tag=tag, process_time_s=process_time_s, data={'moxy-%s-data'%device_id: data})
          self._previous_counters[device_id] = counter
      else:
        print("Unknown data type '%s': %r", data_type, data, flush=True)
    except queue.Empty:
      if not self._is_continue_capture:
        self._send_end_packet()


  def _stop_new_data(self):
    # Stop Ant node from adding new data.
    self.node.stop()


  def _cleanup(self) -> None:
    super()._cleanup()
