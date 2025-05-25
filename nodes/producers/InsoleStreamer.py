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
from streams import InsoleStream

from utils.print_utils import *
from utils.zmq_utils import *
import socket


##################################################
##################################################
# A class to inteface with Moticon insole sensors.
##################################################
##################################################
class InsoleStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'insoles'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               sampling_rate_hz: float = 100,
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
  def create_stream(cls, stream_info: dict) -> InsoleStream:
    return InsoleStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.settimeout(10)
    return True


  def _keep_samples(self) -> None:
    # Bind the socket after nodes synced, ensures no buffering on the socket happens. 
    self._sock.bind((IP_LOOPBACK, int(PORT_MOTICON)))


  def _process_data(self) -> None:
    if self._is_continue_capture:
      try:
        payload, address = self._sock.recvfrom(1024) # data is whitespace-separated byte string
      except socket.timeout:
        print('Moticon insoles receive socket timed out on receive.', flush=True)
        return

      process_time_s: float = get_time()
      payload = [float(word) for word in payload.split()] # splits byte string into array of (multiple) bytes, removing whitespace separators between measurements

      data = {
        'timestamp': payload[0],
        'toa_s': process_time_s,
        'foot_pressure_left': payload[9:25],
        'foot_pressure_right': payload[34:50],
        'acc_left': payload[1:4],
        'acc_right': payload[26:29],
        'gyro_left': payload[4:7],
        'gyro_right': payload[29:32],
        'total_force_left': payload[25],
        'total_force_right': payload[50],
        'center_of_pressure_left': payload[7:9],
        'center_of_pressure_right': payload[32:34],
      }

      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data={'insoles-data': data})
    else:
      self._send_end_packet()


  def _stop_new_data(self):
    self._sock.close()


  def _cleanup(self) -> None:
    super()._cleanup()
