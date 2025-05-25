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
from streams import ExperimentControlStream

from utils.print_utils import *
from utils.zmq_utils import *


#####################################################################
#####################################################################
# A class to create a GUI that can record experiment events.
# Includes calibration periods, activities, and arbitrary user input.
# TODO: allow it to work from CLI, without dependence on GUI.
#####################################################################
#####################################################################
class ExperimentControlStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'control'

  
  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               activities: list[str],
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               **_):
    
    stream_info = {
      "activities": activities
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig)


  @classmethod
  def create_stream(cls, stream_info: dict) -> ExperimentControlStream:
    return ExperimentControlStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  # Connect to the sensor device(s).
  def _connect(self) -> bool:
    return True


  # Clean up and quit
  def _cleanup(self) -> None:
    super()._cleanup()
