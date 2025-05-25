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
from streams import DummyStream

from utils.print_utils import *
from utils.zmq_utils import *
import time


class DummyProducer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'dummy-producer'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               sampling_rate_hz: int = 1,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = float('nan'),
               **_):
    
    stream_info = {
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
  def create_stream(cls, stream_info: dict) -> DummyStream:
    return DummyStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    return True


  def _keep_samples(self) -> None:
    pass


  def _process_data(self) -> None:
    if self._is_continue_capture:
      time.sleep(1.0)
      process_time_s: float = get_time()
      print(process_time_s, flush=True)
      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data={'sensor-emulator': {'toa': process_time_s}})
    else:
      self._send_end_packet()


  def _stop_new_data(self):
    pass


  def _cleanup(self) -> None:
    super()._cleanup()
