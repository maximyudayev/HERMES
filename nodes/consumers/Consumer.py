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

import threading
from handlers.LoggingHandler import Logger
from nodes.Node import Node
from nodes.producers.Producer import Producer
from nodes.pipelines import PIPELINES
from nodes.producers import PRODUCERS
from streams import Stream

from abc import abstractmethod
from collections import OrderedDict
import zmq

from utils.msgpack_utils import deserialize
from utils.zmq_utils import *


##########################################################
##########################################################
# An abstract class to interface with a particular worker.
#   I.e. a superclass for a data logger or passive GUI.
#   Does not by default log consumed data.
##########################################################
##########################################################
class Consumer(Node):
  def __init__(self,
               host_ip: str,
               stream_specs: list[dict],
               logging_spec: dict,
               port_sub: str = PORT_FRONTEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               log_history_filepath: str = None,
               print_status: bool = True,
               print_debug: bool = False) -> None:
    super().__init__(host_ip=host_ip,
                     port_sync=port_sync, 
                     port_killsig=port_killsig, 
                     print_status=print_status, 
                     print_debug=print_debug)
    self._port_sub = port_sub
    self._log_history_filepath = log_history_filepath

    self._is_producer_ended: OrderedDict[str, bool] = OrderedDict()
    self._poll_data_fn = self._poll_data_packets

    # Instantiate all desired Streams that DataLogger will subscribe to.
    self._streams: OrderedDict[str, Stream] = OrderedDict()
    for stream_spec in stream_specs:
      class_name: str = stream_spec['class']
      class_args = stream_spec.copy()
      del(class_args['class'])
      # Create the class object.
      class_type: type[Producer] = {**PRODUCERS,**PIPELINES}[class_name]
      class_object: Stream = class_type.create_stream(class_type, class_args)
      # Store the streamer object.
      self._streams.setdefault(class_type._log_source_tag(), class_object)
      self._is_producer_ended.setdefault(class_type._log_source_tag(), False)

    # Create the DataLogger object
    self._logger = Logger(self._log_source_tag(), **logging_spec)
    # Launch datalogging thread with reference to the Stream object.
    self._logger_thread = threading.Thread(target=self._logger, args=(self._streams,))
    self._logger_thread.start()


  # Initialize backend parameters specific to Consumer.
  def _initialize(self):
    super()._initialize()
    # Socket to subscribe to SensorStreamers
    self._sub: zmq.SyncSocket = self._ctx.socket(zmq.SUB)
    self._sub.connect("tcp://%s:%s" % (DNS_LOCALHOST, self._port_sub))
    
    # Subscribe to topics for each mentioned local and remote streamer
    for tag in self._streams.keys():
      self._sub.subscribe(tag)


  # Launch data receiving.
  def _activate_data_poller(self) -> None:
    self._poller.register(self._sub, zmq.POLLIN)


  # Process custom event first, then Node generic (killsig).
  def _on_poll(self, poll_res):
    if self._sub in poll_res[0]:
      self._poll_data_fn()
    super()._on_poll(poll_res)


  def _on_sync_complete(self) -> None:
    pass


  # In normal operation mode, all messages are 2-part.
  def _poll_data_packets(self) -> None:
    topic, payload = self._sub.recv_multipart()
    msg = deserialize(payload)
    topic_tree: list[str] = topic.decode('utf-8').split('.')
    self._streams[topic_tree[0]].append_data(**msg)


  # When system triggered a safe exit, Consumer gets a mix of normal 2-part messages
  #   and 3-part 'END' message from each Producer that safely exited.
  #   It's more efficient to dynamically switch the callback instead of checking every message.
  def _poll_ending_data_packets(self) -> None:
    # Process until all data sources sent 'END' packet.
    topic, payload = self._sub.recv_multipart()
    # 'END' empty packet from a Producer.
    if CMD_END.encode('utf-8') in payload:
      topic_tree: list[str] = topic.decode('utf-8').split('.')
      self._is_producer_ended[topic_tree[0]] = True
      if all(list(self._is_producer_ended.values())):
        self._is_done = True
    # Regular data packets.
    else:
      msg = deserialize(payload)
      topic_tree: list[str] = topic.decode('utf-8').split('.')
      self._streams[topic_tree[0]].append_data(**msg)


  def _trigger_stop(self):
    self._poll_data_fn = self._poll_ending_data_packets


  # Stop all the data logging.
  # Will stop stream-logging if it is active.
  # Will dump all data if desired.
  @abstractmethod
  def _cleanup(self):
    # Finish up the file saving before exitting.
    self._logger.cleanup()
    self._logger_thread.join()
    # Before closing the PUB socket, wait for the 'BYE' signal from the Broker.
    self._sync.send_multipart([self._log_source_tag().encode('utf-8'), CMD_EXIT.encode('utf-8')]) 
    host, cmd = self._sync.recv_multipart() # no need to read contents of the message.
    print("%s received %s from %s." % (self._log_source_tag(),
                                       cmd.decode('utf-8'),
                                       host.decode('utf-8')),
                                       flush=True)
    self._sub.close()
    super()._cleanup()
