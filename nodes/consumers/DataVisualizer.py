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

from nodes.consumers.Consumer import Consumer

import threading
from wsgiref.simple_server import make_server
import dash_bootstrap_components as dbc

from utils.gui_utils import server, app
from utils.print_utils import *
from utils.zmq_utils import *


######################################
######################################
# A class to visualize streaming data.
######################################
######################################
class DataVisualizer(Consumer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'visualizer'


  def __init__(self,
               host_ip: str,
               stream_specs: list[dict],
               logging_spec: dict,
               log_history_filepath: str | None = None,
               port_sub: str = PORT_FRONTEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               **_):

    super().__init__(host_ip=host_ip,
                     stream_specs=stream_specs,
                     logging_spec=logging_spec,
                     port_sub=port_sub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     log_history_filepath=log_history_filepath)

    # Init all Dash widgets before launching the server and the GUI thread.
    # NOTE: order Dash widgets in the order of streamer specs provided upstream.
    app.layout = dbc.Container([
      visualizer for visualizer in [stream.build_visulizer() for stream in self._streams.values()] if visualizer is not None
    ])

    # Launch Dash GUI thread.
    self._flask_server = make_server(DNS_LOCALHOST, int(PORT_GUI), server)
    self._flask_server_thread = threading.Thread(target=self._flask_server.serve_forever)
    self._flask_server_thread.start()

    self._dash_app_thread = threading.Thread(target=app.run, kwargs={'debug': True, 'use_reloader': False})
    self._dash_app_thread.start()


  def _cleanup(self):
    self._flask_server.shutdown()
    self._flask_server_thread.join()
    self._dash_app_thread.join()
    super()._cleanup()
