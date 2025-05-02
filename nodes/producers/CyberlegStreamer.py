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
from streams import CyberlegStream

from utils.print_utils import *
from utils.zmq_utils import *
from utils.time_utils import get_time
import socket
import struct


######################################################
######################################################
# A class to inteface with AidWear Cyberleg to receive 
#   smartphone activity selection data.
# At the moment consists of 3 bytes:
#   - 0-9 - main task
#   - 0-9 - subtask
#   - o/f - boolean for both motors ON
######################################################
######################################################
class CyberlegStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'cyberleg'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               print_status: bool = True, 
               print_debug: bool = False,
               **_):

    self._num_packet_bytes = 3
    stream_info = {
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     print_status=print_status,
                     print_debug=print_debug)


  def create_stream(cls, stream_info: dict) -> CyberlegStream:
    return CyberlegStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    try:
      self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._sock.connect((IP_PROSTHESIS, int(PORT_PROSTHESIS)))
      self._sock.recv(self._num_packet_bytes)
      return True
    except socket.timeout:
      return False


  def _keep_samples(self) -> None:
    pass


  def _process_data(self) -> None:
    if self._is_continue_capture:
      payload = self._sock.recv(self._num_packet_bytes) 
      process_time_s: float = get_time()
      # Interpret smartphone bytes correctly from the prosthesis.
      msg = struct.unpack('ccc', payload) # TODO: unpack the packet.
      data = {
        'activity':     msg[0],
        'subactivity':  msg[1],
        'motor_state':  msg[2], 
        # 'timestamp':    msg[3],
      }
      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data={'cyberleg-data': data})
    else:
      self._send_end_packet()


  def _stop_new_data(self):
    self._sock.close()


  def _cleanup(self) -> None:
    super()._cleanup()
