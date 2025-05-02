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

from abc import ABC, abstractmethod

import zmq
from utils.zmq_utils import *


class NodeInterface(ABC):
  # Read-only property that every subclass must implement.
  @classmethod
  @abstractmethod
  def _log_source_tag(cls) -> str:
    pass

  @property
  @abstractmethod
  def _is_done(self) -> bool:
    pass

  @abstractmethod
  def _set_state(self, state) -> None:
    pass

  @abstractmethod
  def _initialize(self) -> None:
    pass

  @abstractmethod
  def _get_sync_socket(self) -> zmq.SyncSocket:
    pass

  @abstractmethod
  def _activate_kill_poller(self) -> None:
    pass

  @abstractmethod
  def _activate_data_poller(self) -> None:
    pass

  @abstractmethod
  def _deactivate_kill_poller(self) -> None:
    pass

  @abstractmethod
  def _send_kill_to_broker(self) -> None:
    pass

  @abstractmethod
  def _poll(self) -> tuple[list[zmq.SyncSocket], list[int]]:
    pass

  @abstractmethod
  def _on_poll(self, poll_res: tuple[list[zmq.SyncSocket], list[int]]) -> None:
    pass

  @abstractmethod
  def _trigger_stop(self) -> None:
    pass

  @abstractmethod
  def _on_sync_complete(self) -> None:
    pass


class NodeState(ABC):
  def __init__(self, context: NodeInterface):
    self._context = context

  @abstractmethod
  def run(self) -> None:
    pass

  def is_continue(self) -> bool:
    return True

  def kill(self) -> None:
    self._context._set_state(KillState(self._context))


class StartState(NodeState):
  def run(self):
    self._context._initialize()
    # Activate data poller in case Node goes into KillState.
    self._context._activate_data_poller()
    self._context._set_state(SyncState(self._context))


class SyncState(NodeState):
  def __init__(self, context: NodeInterface):
    super().__init__(context)
    self._sync = context._get_sync_socket()

  def run(self):
    self._sync.send_multipart([self._context._log_source_tag().encode('utf-8'), CMD_HELLO.encode('utf-8')])
    host, cmd = self._sync.recv_multipart()
    print("%s received %s from %s." % (self._context._log_source_tag(),
                                       cmd.decode('utf-8'),
                                       host.decode('utf-8')),
                                       flush=True)
    self._context._set_state(RunningState(self._context))


class RunningState(NodeState):
  def __init__(self, context):
    super().__init__(context)
    self._context._activate_kill_poller()
    self._context._on_sync_complete()

  def run(self):
    poll_res: tuple[list[zmq.SyncSocket], list[int]] = self._context._poll()
    self._context._on_poll(poll_res)


class KillState(NodeState):
  def run(self):
    self._context._deactivate_kill_poller()
    self._context._send_kill_to_broker()
    self._context._trigger_stop()
    self._context._set_state(JoinState(self._context))

  # Override to ignore more kill calls because we are already ending process.
  def kill(self):
    pass


class JoinState(NodeState):
  def run(self):
    poll_res: tuple[list[zmq.SyncSocket], list[int]] = self._context._poll()
    self._context._on_poll(poll_res)

  def is_continue(self):
    return not self._context._is_done
  
  # Override to ignore more kill calls because we are already ending process.
  def kill(self):
    pass


class Node(NodeInterface):
  def __init__(self,
               host_ip: str = DNS_LOCALHOST,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               print_status: bool = True,
               print_debug: bool = False) -> None:
    self._print_status = print_status
    self._print_debug = print_debug
    self._host_ip = host_ip
    self._port_sync = port_sync
    self._port_killsig = port_killsig
    self.__is_done = False

    self._state = StartState(self)

    self._ctx: zmq.Context = zmq.Context.instance()
    self._poller: zmq.Poller = zmq.Poller()


  @property
  def _is_done(self) -> str:
    return self.__is_done
  

  @_is_done.setter
  def _is_done(self, done: bool) -> None:
    self.__is_done = done


  # Nodes are callable with FSM as entry-point.
  def __call__(self):
    while self._state.is_continue():
      self._state.run()
    self._cleanup()
    print("%s exited, goodbye <3"%self._log_source_tag(), flush=True)


  # FSM transition.
  def _set_state(self, state: NodeState) -> None:
    self._state = state


  # Pre-run setup of the backend specific to the Node implementaiton.
  # Generic setup should be run first.
  @abstractmethod
  def _initialize(self):
    # Socket to receive kill signal
    self._killsig: zmq.SyncSocket = self._ctx.socket(zmq.SUB)
    self._killsig.connect("tcp://%s:%s" % (DNS_LOCALHOST, self._port_killsig))
    topics = [TOPIC_KILL]
    for topic in topics: self._killsig.subscribe(topic)

    # Socket to indicate to broker that the subscriber is ready
    self._sync: zmq.SyncSocket = self._ctx.socket(zmq.REQ)
    self._sync.connect("tcp://%s:%s" % (self._host_ip, self._port_sync))
    # Socket to indicate to broker that the Node caught interrupt signal
    self._babykillsig: zmq.SyncSocket = self._ctx.socket(zmq.REQ)
    self._babykillsig.connect("tcp://%s:%s" % (DNS_LOCALHOST, PORT_KILL_BTN))


  def _get_sync_socket(self) -> zmq.SyncSocket:
    return self._sync


  # Start listening to the kill signal
  def _activate_kill_poller(self) -> None:
    self._poller.register(self._killsig, zmq.POLLIN)


  @abstractmethod
  def _activate_data_poller(self) -> None:
    pass


  # Stop listening to the kill signal.
  def _deactivate_kill_poller(self) -> None:
    print("%s received KILL signal"%self._log_source_tag(), flush=True)
    # self._killsig.recv_multipart()
    self._poller.unregister(self._killsig)


  def _send_kill_to_broker(self):
    self._babykillsig.send_string(TOPIC_KILL)


  # Listens for events when new data is received from or when new data can be written to sockets,
  #   based on the active Poller settings of the Node implementation.
  def _poll(self) -> tuple[list[zmq.SyncSocket], list[int]]:
    return tuple(zip(*(self._poller.poll())))


  # Actions to perform on the poll event.
  # Generic entry-point for all types of Nodes, based on their active Poller settings.
  # NOTE: if Node in JoinState, kill socket is no longer in the Poller and only higher-level logic is triggered.
  @abstractmethod
  def _on_poll(self, poll_res: tuple[list[zmq.SyncSocket], list[int]]) -> None:
    if self._killsig in poll_res[0]:
      self._state.kill()


  # Send signal to inheriting Node type to cooldown.
  #   Producer: stops sampling data, continue sending already captured until none is left, with last message labeled 'END'.
  #   Consumer: continues listening to data until each of subscribed Producers sent the last message.
  #   Pipeline: continues listening to data to produce results until each data sources sent the last message, and then labels the last message with 'END'.
  @abstractmethod
  def _trigger_stop(self) -> None:
    pass


  # Release of generic Node resources, must be done after releasing higher-level resources.
  @abstractmethod
  def _cleanup(self):
    self._killsig.close()
    self._babykillsig.close()
    self._sync.close()
    # Destroy ZeroMQ context.
    self._ctx.term()
