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
from streams.CometaStream import CometaStream

from handlers.CometaWaveplus.CometaHandler import CometaFacade
from utils.zmq_utils import *

from utils.time_utils import get_time


######################################
######################################
# A class for streaming Dots IMU data.
######################################
######################################
class CometaStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'emgs'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               device_mapping: dict[str, str],
               sampling_rate_hz: int = 2000,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               timesteps_before_solidified: int = 0,
               **_):

    # Initialize any state that the sensor needs.
    self._device_mapping = device_mapping
    self._sampling_rate_hz = sampling_rate_hz

    stream_info = {
      "device_mapping": device_mapping,
      "sampling_rate_hz": sampling_rate_hz,
      "timesteps_before_solidified": timesteps_before_solidified
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     sampling_rate_hz=sampling_rate_hz,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig)


  @classmethod
  def create_stream(cls, stream_info: dict) -> CometaStream:
    return CometaStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    self._handler = CometaFacade(device_mapping=self._device_mapping)
    # Keep reconnecting until success
    while not self._handler.initialize():
      self._handler.cleanup()
    return True


  def _keep_samples(self) -> None:
    self._handler.keep_data()


  def _process_data(self) -> None:
    # Retrieve the oldest enqueued packet for each sensor.
    snapshot = self._handler.get_packet()
    if snapshot is not None:
      process_time_s: float = get_time()
      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data=snapshot)
    elif not self._is_continue_capture:
      # If triggered to stop and no more available data, send empty 'END' packet and join.
      self._send_end_packet()


  def _stop_new_data(self):
    self._handler.cleanup()


  def _cleanup(self) -> None:
    self._handler.close()
    super()._cleanup()
