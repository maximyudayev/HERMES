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
from multiprocessing import Process, set_start_method
from typing import Callable

import zmq

from utils.node_utils import launch_node
from utils.time_utils import *
from utils.dict_utils import *
from utils.print_utils import *
from utils.types import ZMQResult
from utils.zmq_utils import *


################################################################################
################################################################################
# PUB-SUB Broker to manage a collection of Node objects.
# Hosts control logic of interactive proxy/server.
# Will launch/destroy/connect to streamers on creation and ad-hoc.
# Will use a separate process for each streamer and consumer.
# Will use the main process to:
#   * route PUB-SUB messages;
#   * manage lifecycle of locally-connected Nodes;
#   * TODO: subscribe to stdin/stdout messages of all publishers and subscribers
# Each Node connects only to its local broker,
#   which then exposes its data to outside LAN subscribers.
################################################################################
################################################################################
class BrokerInterface(ABC):
  # Read-only property that every subclass must implement.
  @classmethod
  @abstractmethod
  def _log_source_tag(cls) -> str:
    pass

  @abstractmethod
  def _start_local_nodes(self) -> None:
    pass

  @abstractmethod
  def _set_state(self, state) -> None:
    pass

  @abstractmethod
  def _get_num_local_nodes(self) -> int:
    pass

  @abstractmethod
  def _get_num_frontends(self) -> int:
    pass
  
  @abstractmethod
  def _get_num_backends(self) -> int:
    pass

  @abstractmethod
  def _get_remote_pub_brokers(self) -> list[str]:
    pass
  
  @abstractmethod
  def _get_remote_sub_brokers(self) -> list[str]:
    pass

  @abstractmethod
  def _get_is_master_broker(self) -> bool:
    pass

  @abstractmethod
  def _get_brokered_nodes(self) -> set[str]:
    pass

  @abstractmethod
  def _add_brokered_node(self, topic: str) -> None:
    pass

  @abstractmethod
  def _remove_brokered_node(self, topic: str) -> None:
    pass

  @abstractmethod
  def _get_start_time(self) -> float:
    pass

  @abstractmethod
  def _get_duration(self) -> float | None:
    pass

  @abstractmethod
  def _get_sync_host_socket(self) -> zmq.SyncSocket:
    pass

  @abstractmethod
  def _get_sync_remote_socket(self) -> zmq.SyncSocket:
    pass

  @abstractmethod 
  def _set_node_addresses(self, nodes: dict[str, bytes]) -> None:
    pass

  @abstractmethod 
  def _set_remote_broker_addresses(self, remote_brokers: dict[str, bytes]) -> None:
    pass

  @abstractmethod
  def _get_remote_broker_addresses(self) -> dict[str, bytes]:
    pass

  @abstractmethod
  def _get_host_ip(self) -> str:
    pass

  @abstractmethod
  def _get_node_addresses(self) -> dict[str, bytes]:
    pass

  @abstractmethod
  def _activate_pubsub_poller(self) -> None:
    pass

  @abstractmethod
  def _deactivate_pubsub_poller(self) -> None:
    pass

  @abstractmethod
  def _get_poller(self) -> zmq.Poller:
    pass

  @abstractmethod
  def _poll(self, timeout_ms: int) -> ZMQResult:
    pass

  @abstractmethod
  def _broker_packets(self,
                      poll_res: ZMQResult,
                      on_data_received: Callable[[list[bytes]], None] = lambda _: None,
                      on_subscription_changed: Callable[[list[bytes]], None] = lambda _: None) -> None:
    pass

  @abstractmethod
  def _check_for_kill(self, poll_res: ZMQResult) -> bool:
    pass

  @abstractmethod
  def _publish_kill(self):
    pass


class BrokerState(ABC):
  def __init__(self, context: BrokerInterface):
    self._context = context

  @abstractmethod
  def run(self) -> None:
    pass

  def is_continue(self) -> bool:
    return True

  def kill(self) -> None:
    self._context._set_state(KillState(self._context))


# Activates broker poller sockets and goes in sync to wait for local Nodes to start up.
class InitState(BrokerState):
  def run(self) -> None:
    self._context._activate_pubsub_poller()
    self._context._start_local_nodes()
    self._context._set_state(SyncNodeBarrierState(self._context))


# Waits until all local Nodes signalled that they are initialized and ready to go.
class SyncNodeBarrierState(BrokerState):
  def run(self) -> None:
    host_ip: str = self._context._get_host_ip()
    sync_host_socket: zmq.SyncSocket = self._context._get_sync_host_socket()
    num_left_to_sync: int = self._context._get_num_local_nodes()
    nodes = dict()
    while num_left_to_sync:
      address, _, node_name, cmd = sync_host_socket.recv_multipart()
      num_left_to_sync -= 1
      node_name = node_name.decode('utf-8')
      nodes[node_name] = address
      print("%s connected to %s with %s message." % (node_name,
                                                     host_ip,
                                                     cmd.decode('utf-8')), flush=True)
    self._context._set_node_addresses(nodes)
    self._context._set_state(SyncBrokerBarrierState(self._context))


# Communicate to other Brokers that every device is ready.
class SyncBrokerBarrierState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    self._host_ip: str = self._context._get_host_ip()
    self._sync_remote_socket: zmq.SyncSocket = self._context._get_sync_remote_socket()

    self._remote_sub_brokers = self._context._get_remote_sub_brokers()
    self._remote_pub_brokers = self._context._get_remote_pub_brokers()
    self._brokers = dict()

    self._brokers_left_to_acknowledge = set(self._remote_sub_brokers)
    self._brokers_left_to_checkin = set(self._remote_pub_brokers)
    for ip in self._remote_sub_brokers:
      self._sync_remote_socket.connect('tcp://%s:%s'%(ip, PORT_SYNC_REMOTE))
    # Register remote SYNC socket to receive requests from remote publishers,
    #   and responses from remote subscribers.
    self._poller = zmq.Poller()
    self._poller.register(self._sync_remote_socket, zmq.POLLIN)


  def run(self) -> None:
    # I am remote publishing Broker, I must notify subscribing Brokers that I am ready.
    for ip in self._remote_sub_brokers:
      self._sync_remote_socket.send_multipart([("%s:%s"%(ip, PORT_SYNC_REMOTE)).encode('utf-8'),
                                               b'', 
                                               self._host_ip.encode('utf-8'), 
                                               CMD_HELLO.encode('utf-8')])

    # Check every 5 seconds if other Brokers completed their setup and responded back.
    # Could be that no other Brokers exist.
    poll_res: list[tuple[zmq.SyncSocket, zmq.PollEvent]]
    if poll_res := self._poller.poll(5000): # type: ignore
      socket, _ = poll_res[0]
      address, _, broker_name, cmd = socket.recv_multipart()
      broker_name = broker_name.decode('utf-8')
      print("%s sent %s to %s response" % (broker_name,
                                           cmd.decode('utf-8'),
                                           self._host_ip),
                                           flush=True)
      if broker_name in self._brokers_left_to_acknowledge:
        # Remote publisher received ACK from remote subscriber.
        self._brokers_left_to_acknowledge.remove(broker_name)
        self._brokers[broker_name] = address
      elif broker_name in self._brokers_left_to_checkin:
        self._brokers_left_to_checkin.remove(broker_name)
        self._brokers[broker_name] = address
        # Remote subscriber responds with ACK to remote publisher.
        self._sync_remote_socket.send_multipart([address,
                                                 b'', 
                                                 self._host_ip.encode('utf-8'), 
                                                 CMD_ACK.encode('utf-8')])

    # Proceed to the next state to agree on the common time once all Brokers synchronized.
    if not self._brokers_left_to_acknowledge and not self._brokers_left_to_checkin:
      self._poller.unregister(self._sync_remote_socket)
      self._context._set_remote_broker_addresses(self._brokers)
      self._context._set_state(StartState(self._context))


# Trigger local Nodes to start logging when the agreed start time arrives.
class StartState(BrokerState):
  def run(self) -> None:
    # If current Broker is not Master, wait for the SYNC signal with time when to start.
    host_ip: str = self._context._get_host_ip()
    sync_remote_socket: zmq.SyncSocket = self._context._get_sync_remote_socket()
    sync_host_socket: zmq.SyncSocket = self._context._get_sync_host_socket()
    nodes: dict[str, bytes] = self._context._get_node_addresses()
    brokers: dict[str, bytes] = self._context._get_remote_broker_addresses()

    # Master Broker selects start time as 5 seconds from now and distributes across Brokers.
    if self._context._get_is_master_broker():
      start_time_s: int = round(get_time()) + 5
      for address in brokers.values():
        sync_remote_socket.send_multipart([address,
                                           b'',
                                           CMD_START_TIME.encode('utf-8'),
                                           start_time_s.to_bytes(length=4, byteorder='big')])
    # Slave Brokers block on the reeceive socket, waiting for the time. 
    else:
      address, _, cmd, start_time_bytes = sync_remote_socket.recv_multipart()
      start_time_s = int.from_bytes(start_time_bytes, byteorder='big')

    # Each Broker waits until that time comes to trigger start of logging, with 1ms precision.
    while (current_time_s := get_time()) < start_time_s:
      time.sleep(min(0.001, start_time_s-current_time_s))
    
    # Trigget local Nodes to start logging.
    for name, address in list(nodes.items()):
      sync_host_socket.send_multipart([address,
                                       b'',
                                       host_ip.encode('utf-8'),
                                       CMD_GO.encode('utf-8')])
      print("%s sending %s to %s" % (host_ip,
                                     CMD_GO,
                                     name),
                                     flush=True)

    self._context._set_state(RunningState(self._context))


# Will run until the the experiment is stopped or after a fixed period, if provided.
class RunningState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    if (duration_s := self._context._get_duration()) is not None:
      self._is_continue_fn = lambda: get_time() < (self._context._get_start_time() + duration_s)
    else:
      self._is_continue_fn = lambda: True


  def run(self) -> None:
    poll_res: ZMQResult = self._context._poll(5000)
    self._context._broker_packets(poll_res, on_subscription_changed=self._on_subscription_added)
    if self._context._check_for_kill(poll_res): self.kill()


  def is_continue(self) -> bool:
    return self._is_continue_fn()
  

  # Update a list on the Broker that keeps track of which Nodes are being brokered for. 
  def _on_subscription_added(self, msg: list[bytes]) -> None:
    topic: str = msg[0].decode('utf-8').split('\x01')[1]
    self._context._add_brokered_node(topic=topic)


# Received the KILL signal, relay it to all Nodes and Brokers and wrap up gracefully.
#   from the local Keyboard Interrupt;
#   from the Master Broker;
#   from the GUI;
class KillState(BrokerState):
  def run(self) -> None:
    self._context._publish_kill()
    self._context._set_state(JoinNodeBarrierState(self._context))


  # Override default kill function behavior because we are already in the killing process
  def kill(self) -> None:
    pass


# Waits until all local Nodes send final packets then quits itself.
class JoinNodeBarrierState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    self._host_ip = self._context._get_host_ip()
    self._nodes = self._context._get_node_addresses()
    self._nodes_waiting_to_exit: set[str] = set()
    self._nodes_expected_end_pub_packet: set[str] = self._context._get_brokered_nodes()
    self._sync_host_socket: zmq.SyncSocket = self._context._get_sync_host_socket()
    self._poller = self._context._get_poller()
    self._poller.register(self._sync_host_socket, zmq.POLLIN)


  # Wait for all processes (local and remote) to send the last messages before closing.
  #   Continue brokering packets until signalled by all publishers that there will be no more packets.
  #   Append a frame to the ZeroMQ message that indicates the last message from the sensor.
  def run(self) -> None:
    poll_res: ZMQResult = self._context._poll(5000)
    # Brokers packets and releases local Producer Nodes in a callback once it published the end packet.
    self._context._broker_packets(poll_res, on_data_received=self._on_is_end_packet)
    # Checks if poll event was triggered by a local Node initiating closing.
    self._check_host_sync_socket(poll_res)
    # Proceed to exiting once all local Nodes finished.
    if self._is_finished():
      self._poller.unregister(self._sync_host_socket)
      self._context._deactivate_pubsub_poller()
      self._context._set_state(JoinBrokerBarrierState(self._context))


  # Callback to track brokering of last packets of local Producers and Pipelines.
  #   Will get trigerred at most once per Node because Nodes send it only once.
  def _on_is_end_packet(self, msg: list[bytes]) -> None:
    #   Once the Broker registers arrival of 'END' packet from a local Producer/Pipeline, 
    #     it will signal 'BYE' to it to allow it to exit.
    if CMD_END.encode('utf-8') in msg:
      # Check if the END packet came from the Broker's scope, (one of the Broker's local Nodes).
      #   Continue brokering packets if just proxing it (not Broker's local Nodes).
      topic = msg[0].decode().split('.')[0]
      if self._nodes_expected_end_pub_packet:
        self._nodes_expected_end_pub_packet.remove(topic)
        # Allow local Producer/Pipeline to exit.
        self._release_local_node(topic)


  def _check_host_sync_socket(self, poll_res: ZMQResult) -> None:
    # Can be triggered by all local Nodes: Producer, Consumer, or Pipeline, sending 'EXIT?' request.
    for sock, _ in poll_res:
      if sock == self._sync_host_socket:
        address, _, node_name, cmd = self._sync_host_socket.recv_multipart()
        topic = node_name.decode('utf-8')
        print("%s received %s from %s" % (self._host_ip,
                                          cmd,
                                          topic),
                                          flush=True)
        self._nodes_waiting_to_exit.add(topic)
        self._release_local_node(topic)


  def _release_local_node(self, topic: str) -> None:
    if topic in self._nodes_waiting_to_exit and topic not in self._nodes_expected_end_pub_packet:
      self._sync_host_socket.send_multipart([self._nodes[topic],
                                             b'',
                                             self._host_ip.encode('utf-8'),
                                             CMD_BYE.encode('utf-8')])
      del(self._nodes[topic])
      self._nodes_waiting_to_exit.remove(topic)


  def _is_finished(self) -> bool:
    return not self._nodes


  def kill(self) -> None:
    pass


class JoinBrokerBarrierState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    self._host_ip = self._context._get_host_ip()
    self._brokers = self._context._get_remote_broker_addresses()
    self._sync_remote_socket: zmq.SyncSocket = self._context._get_sync_remote_socket()
    
    self._remote_sub_brokers = self._context._get_remote_sub_brokers()
    self._remote_pub_brokers = self._context._get_remote_pub_brokers()

    self._brokers_left_to_acknowledge = set(self._remote_sub_brokers)
    self._brokers_left_to_checkin = set(self._remote_pub_brokers)

    self._poller = zmq.Poller()
    self._poller.register(self._sync_remote_socket, zmq.POLLIN)


  def run(self):
    # Notify Brokers that listen to our data that we are done and ready to exit as soon as they received all last data from us.
    for ip in self._remote_sub_brokers:
      self._sync_remote_socket.send_multipart([("%s:%s"%(ip, PORT_SYNC_REMOTE)).encode('utf-8'),
                                               b'', 
                                               self._host_ip.encode('utf-8'),
                                               CMD_HELLO.encode('utf-8')])

    # Check every 5 seconds if other Brokers completed their cleanup and responded back ready to exit.
    poll_res: list[tuple[zmq.SyncSocket, zmq.PollEvent]]
    if poll_res := self._poller.poll(5000): # type: ignore
      socket, _ = poll_res[0]
      address, _, broker_name, cmd = socket.recv_multipart()
      broker_name = broker_name.decode('utf-8')
      print("%s sent %s to %s." % (broker_name,
                                   cmd.decode('utf-8'),
                                   self._host_ip),
                                   flush=True)

      if broker_name in self._brokers_left_to_acknowledge:
        # Remote subscriber responded with ACK to us.
        self._brokers_left_to_acknowledge.remove(broker_name)
        self._brokers.pop(broker_name)
      elif broker_name in self._brokers_left_to_checkin:
        # Remote publisher sent a BYE request, respond with an ACK.
        self._brokers_left_to_checkin.remove(broker_name)
        self._brokers.pop(broker_name)
        self._sync_remote_socket.send_multipart([address,
                                                 b'',
                                                 self._host_ip.encode('utf-8'),
                                                 CMD_BYE.encode('utf-8')])


  def is_continue(self) -> bool:
    return not not self._brokers


  def kill(self) -> None:
    pass


class Broker(BrokerInterface):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'manager'


  # Initializes all broker logic and launches nodes
  def __init__(self,
               host_ip: str,
               node_specs: list[dict],
               port_backend: str = PORT_BACKEND,
               port_frontend: str = PORT_FRONTEND,
               port_sync_host: str = PORT_SYNC_HOST,
               port_sync_remote: str = PORT_SYNC_REMOTE,
               port_killsig: str = PORT_KILL,
               is_master_broker: bool = False) -> None:

    # Record various configuration options.
    self._host_ip = host_ip
    self._is_master_broker = is_master_broker
    self._port_backend = port_backend
    self._port_frontend = port_frontend
    self._port_sync_host = port_sync_host
    self._port_sync_remote = port_sync_remote
    self._port_killsig = port_killsig
    self._node_specs = node_specs
    self._is_quit = False

    self._remote_pub_brokers: list[str] = []
    self._remote_sub_brokers: list[str] = []
    self._brokered_nodes: set[str] = set()

    # FSM for the broker
    self._state = InitState(self)

    ###########################
    ###### CONFIGURATION ######
    ###########################
    # NOTE: We don't want streamers to share memory, each is a separate process communicating and sharing data over sockets
    #   ActionSense used multiprocessing.Manager and proxies to access streamers' data from the main process.
    # NOTE: Lab PC needs to receive packets on 2 interfaces - internally (own loopback) and over the network from the wearable PC.
    #   It then brokers data to its workers (data logging, visualization) in an abstract way so they don't have to know sensor topology.
    # NOTE: Wearable PC needs to send packets on 2 interfaces - internally (own loopback) and over the network to the lab PC.
    #   To wearable PC, lab PC looks just like another subscriber.
    # NOTE: Loopback and LAN can use the same port of the device because different interfaces are treated as independent connections.
    # NOTE: Loopback is faster than the network interface of the same device because it doesn't have to go through routing tables.

    # Pass exactly one ZeroMQ context instance throughout the program
    self._ctx: zmq.Context = zmq.Context()

    # Exposes a known address and port to locally connected sensors to connect to.
    local_backend: zmq.SyncSocket = self._ctx.socket(zmq.XSUB)
    local_backend.bind("tcp://%s:%s" % (IP_LOOPBACK, self._port_backend))
    self._backends: list[zmq.SyncSocket] = [local_backend]

    # Exposes a known address and port to broker data to local workers.
    local_frontend: zmq.SyncSocket = self._ctx.socket(zmq.XPUB)
    local_frontend.bind("tcp://%s:%s" % (IP_LOOPBACK, self._port_frontend))
    self._frontends: list[zmq.SyncSocket] = [local_frontend]

    # Listener endpoint to receive signals of streamers' readiness
    self._sync_host: zmq.SyncSocket = self._ctx.socket(zmq.ROUTER)
    self._sync_host.bind("tcp://%s:%s" % (self._host_ip, self._port_sync_host))

    # Socket to connect to remote Brokers
    self._sync_remote: zmq.SyncSocket = self._ctx.socket(zmq.ROUTER)
    self._sync_remote.setsockopt_string(zmq.IDENTITY, "%s:%s"%(self._host_ip, self._port_sync_remote))
    self._sync_remote.bind("tcp://%s:%s" % (self._host_ip, self._port_sync_remote))

    # Termination control socket to command publishers and subscribers to finish and exit.
    killsig_pub: zmq.SyncSocket = self._ctx.socket(zmq.PUB)
    killsig_pub.bind("tcp://*:%s" % (self._port_killsig))
    self._killsigs: list[zmq.SyncSocket] = [killsig_pub]

    # Socket to listen to kill command from the GUI.
    self._gui_btn_kill: zmq.SyncSocket = self._ctx.socket(zmq.REP)
    self._gui_btn_kill.bind("tcp://*:%s" % (PORT_KILL_BTN))

    # Poll object to listen to sockets without blocking
    self._poller: zmq.Poller = zmq.Poller()


  # Exposes a known address and port to remote networked subscribers if configured.
  def expose_to_remote_broker(self, addr: list[str]) -> None:
    frontend_remote: zmq.SyncSocket = self._ctx.socket(zmq.XPUB)
    frontend_remote.bind("tcp://%s:%s" % (self._host_ip, self._port_frontend))
    self._remote_sub_brokers.extend(addr)
    self._frontends.append(frontend_remote)


  # Connects to a known address and port of external LAN data broker.
  def connect_to_remote_broker(self, addr: str, port_pub: str = PORT_FRONTEND) -> None:
    backend_remote: zmq.SyncSocket = self._ctx.socket(zmq.XSUB)
    backend_remote.connect("tcp://%s:%s" % (addr, port_pub))
    self._remote_pub_brokers.append(addr)
    self._backends.append(backend_remote)


  # Subscribes to external kill signal (e.g. lab PC in AidFOG project).
  def subscribe_to_killsig(self, addr: str, port_killsig: str = PORT_KILL) -> None:
    killsig_sub: zmq.SyncSocket = self._ctx.socket(zmq.SUB)
    killsig_sub.connect("tcp://%s:%s" % (addr, port_killsig))
    killsig_sub.subscribe(TOPIC_KILL)
    self._poller.register(killsig_sub, zmq.POLLIN)
    self._killsigs.append(killsig_sub)


  def set_is_quit(self) -> None:
    self._is_quit = True


  #####################
  ###### RUNNING ######
  #####################
  # The main run method
  #   Runs continuously until the user ends the experiment or after the specified duration.
  #   The duration start to count only after all Nodes established communication and synced.
  def __call__(self, duration_s: float | None = None) -> None:
    self._duration_s = duration_s
    while self._state.is_continue() and not self._is_quit:
      self._state.run()
    if self._is_quit:
      print("Keyboard exit signalled. Safely closing and saving, have some patience...", flush=True)
    self._state.kill()
    # Continue with the FSM until it gracefully wraps up.
    while self._state.is_continue():
      self._state.run()
    self._stop()
    print("Experiment ended, thank you for using our system <3", flush=True)


  #############################
  ###### GETTERS/SETTERS ######
  #############################
  def _set_state(self, state: BrokerState) -> None:
    self._state = state
    self._state_start_time_s = get_time()


  def _set_node_addresses(self, nodes: dict[str, bytes]) -> None:
    self._nodes = nodes


  def _get_node_addresses(self) -> dict[str, bytes]:
    return self._nodes


  def _set_remote_broker_addresses(self, remote_brokers: dict[str, bytes]) -> None:
    self._remote_brokers = remote_brokers


  def _get_remote_broker_addresses(self) -> dict[str, bytes]:
    return self._remote_brokers


  # Start time of the current state - useful for measuring run time of the experiment, excluding the lengthy setup process
  def _get_start_time(self) -> float:
    return self._state_start_time_s


  # User-requested run time of the experiment 
  def _get_duration(self) -> float | None:
    return self._duration_s


  def _get_num_local_nodes(self) -> int:
    return len(self._processes)


  def _get_num_frontends(self) -> int:
    return len(self._frontends)


  def _get_num_backends(self) -> int:
    return len(self._backends)


  def _get_remote_pub_brokers(self) -> list[str]:
    return self._remote_pub_brokers


  def _get_remote_sub_brokers(self) -> list[str]:
    return self._remote_sub_brokers


  def _get_is_master_broker(self) -> bool:
    return self._is_master_broker


  def _get_brokered_nodes(self) -> set[str]:
    return self._brokered_nodes


  def _add_brokered_node(self, topic: str) -> None:
    self._brokered_nodes.add(topic)


  def _remove_brokered_node(self, topic: str) -> None:
    self._brokered_nodes.remove(topic)


  def _get_host_ip(self) -> str:
    return self._host_ip


  # Reference to the RCV socket for syncing
  def _get_sync_host_socket(self) -> zmq.SyncSocket:
    return self._sync_host
  
  
  def _get_sync_remote_socket(self) -> zmq.SyncSocket:
    return self._sync_remote


  def _get_poller(self) -> zmq.Poller:
    return self._poller


  # Register PUB-SUB sockets on both interfaces for polling.
  def _activate_pubsub_poller(self) -> None:
    for s in self._backends:
      self._poller.register(s, zmq.POLLIN)
    for s in self._frontends:
      self._poller.register(s, zmq.POLLIN)
    # Register KILL_BTN port REP socket with POLLIN event.
    self._poller.register(self._gui_btn_kill, zmq.POLLIN)


  def _deactivate_pubsub_poller(self) -> None:
    for s in self._backends:
      self._poller.unregister(s)
    for s in self._frontends:
      self._poller.unregister(s)


  # Spawn local producers and consumers in separate processes
  def _start_local_nodes(self) -> None:
    # Make sure that the child processes are spawned and not forked.
    set_start_method('spawn')
    # Start each publisher-subscriber in its own process (e.g. local sensors, data logger, visualizer, AI worker).
    self._processes: list[Process] = [Process(target=launch_node,
                                              args=(spec,
                                                    self._host_ip,
                                                    self._port_backend,
                                                    self._port_frontend,
                                                    self._port_sync_host,
                                                    self._port_killsig)) for spec in self._node_specs]
    for p in self._processes: p.start()


  # Block until new packets are available.
  def _poll(self, timeout_ms: int) -> ZMQResult:
    return self._poller.poll(timeout=timeout_ms)


  # Move packets between publishers and subscribers.
  def _broker_packets(self, 
                      poll_res: ZMQResult,
                      on_data_received: Callable[[list[bytes]], None] = lambda _: None,
                      on_subscription_changed: Callable[[list[bytes]], None] = lambda _: None) -> None:
    for recv_socket, _ in poll_res:
      # Forwards data packets from publishers to subscribers.
      if recv_socket in self._backends:
        msg = recv_socket.recv_multipart()
        on_data_received(msg)
        for send_socket in self._frontends:
          send_socket.send_multipart(msg)
      # Forwards subscription packets from subscribers to publishers.
      if recv_socket in self._frontends:
        msg = recv_socket.recv_multipart()
        on_subscription_changed(msg)
        for send_socket in self._backends:
          send_socket.send_multipart(msg)


  # Check if packets contain a kill signal from downstream a broker
  def _check_for_kill(self, poll_res: ZMQResult) -> bool:
    for sock, _ in poll_res:
      # Receives KILL from the GUI.
      if sock == self._gui_btn_kill:
        return True
      # Receives KILL signal from another broker.
      elif sock in self._killsigs:
        return True
    return False


  # Send kill signals to upstream brokers and local publishers
  def _publish_kill(self) -> None:
    for kill_socket in self._killsigs[1:]:
      # Ignore any more KILL signals, enter the wrap-up routine.
      self._poller.unregister(kill_socket)
    # Ignore poll events from the GUI and the same socket if used by child processes to indicate keyboard interrupt.
    self._poller.unregister(self._gui_btn_kill)
    # Send kill signals to own locally connected devices.
    self._killsigs[0].send(TOPIC_KILL.encode('utf-8'))


  def _stop(self) -> None:
    # Wait for all the local subprocesses to gracefully exit before terminating the main process.
    for p in self._processes: p.join()

    # Release all used local sockets.
    for s in self._backends: s.close()
    for s in self._frontends: s.close()
    for s in self._killsigs: s.close()
    self._sync_host.close()
    self._sync_remote.close()
    self._gui_btn_kill.close()

    # Destroy ZeroMQ context.
    self._ctx.term()
