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


# The implementation was largely adapted from https://github.com/delpreto/ActionNet


# ############
#
# Copyright (c) 2022 MIT CSAIL and Joseph DelPreto
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
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# See https://action-net.csail.mit.edu for more usage information.
# Created 2021-2022 for the MIT ActionNet project by Joseph DelPreto [https://josephdelpreto.com].
#
############


from enum import Enum, IntEnum
import socket
import struct
from typing import Any, TypeAlias, TypedDict
import numpy as np

from nodes.producers.Producer import Producer
from streams import MvnAnalyzeStream
from streams.MvnAnalyzeStream import MVN_JOINT_SETUP, MVN_SEGMENT_SETUP, MVN_SENSOR_SETUP
from utils.dict_utils import *
from utils.print_utils import *
from utils.time_utils import *
from utils.angle_utils import *
from utils.types import NewDataDict
from utils.zmq_utils import *


class MvnNetProto(str, Enum):
  UDP = 'udp'
  TCP = 'tcp'

class MvnMsgType(IntEnum):
  POSE_EULER          = 1
  POSE_QUATERNION     = 2
  JOINT_ANGLES        = 20
  LINEAR_SEGMENTS     = 21
  ANGULAR_SEGMENTS    = 22
  MOTION_TRACKERS     = 23
  CENTER_OF_MASS      = 24
  TIME_CODE           = 25

MvnMetadata = TypedDict('MvnMetadata', {
  'message_type':       int | None,
  'counter':            int | None,
  'datagram_counter':   int | None,
  'num_items':          int | None,
  'time_since_start_s': int | None,
  'char_id':            int | None,
  'num_segments':       int | None,
  'num_props':          int | None,
  'num_fingers':        int | None,
  'reserved':           int | None,
  'payload_size':       int | None})


def MvnMetadataEmpty() -> MvnMetadata:
  return MvnMetadata(message_type=None,
                     counter=None,
                     datagram_counter=None,
                     num_items=None,
                     time_since_start_s=None,
                     char_id=None,
                     num_segments=None,
                     num_props=None,
                     num_fingers=None,
                     reserved=None,
                     payload_size=None)
    

MvnDataTuple: TypeAlias = tuple[dict[str, Any], int | None]
MvnCommonData: TypeAlias = dict[str, Any]

MVN_UDP_ID_STRING = "MXTP".encode('utf-8')

##############################################################################
##############################################################################
# A class for MVN Analyze Network Streamer data.
# MVN sends single measurement in multiple packets.
#  The host time of the first is used for each packet in a single measurement.
# Streams the following data:
#   * Quaternion orientation data
#   * Euler pose and orientation data
#   * Joint angles
#   * Center of mass dynamics
#   * Raw inertial tracker data
#   * Device timestamps for each timestep
# NOTE: consider WiFi and BLE coexistence. Awinda is also 802.15.4 PHY.
#       Most likely non-coliding channels are 11, 15, 20 or 25.
##############################################################################
##############################################################################
class MvnAnalyzeStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'mvn-analyze'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               mvn_ip: str = IP_LOOPBACK,
               mvn_port: str = PORT_MVN,
               mvn_protocol: MvnNetProto | str = MvnNetProto.UDP,
               mvn_setup: str = "full_body",
               sampling_rate_hz: int = 60,
               is_euler: bool = False,
               is_quaternion: bool = False,
               is_joint_angles: bool = False,
               is_linear_segments: bool = False,
               is_angular_segments: bool = False,
               is_motion_trackers: bool = False,
               is_com: bool = False,
               is_time_code: bool = False,
               buffer_read_size: int = 2048,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = float('nan'),
               **_):

    self._mvn_segment_setup = MVN_SEGMENT_SETUP[mvn_setup]
    self._mvn_joint_setup = MVN_JOINT_SETUP[mvn_setup]
    self._mvn_sensor_setup = MVN_SENSOR_SETUP[mvn_setup]

    self._segment_id_mapping: dict[int, int] = dict(zip(self._mvn_segment_setup.keys(), range(len(self._mvn_segment_setup))))
    self._joint_id_mapping: dict[int, int] = dict(zip(self._mvn_joint_setup.keys(), range(len(self._mvn_joint_setup))))
    self._sensor_id_mapping: dict[int, int] = dict(zip(self._mvn_sensor_setup.keys(), range(len(self._mvn_sensor_setup))))

    # Initialize counts of segments/joints/fingers.
    # These will be updated automatically later based on initial streaming data.
    self._num_segments = len(self._mvn_segment_setup)
    self._num_sensors = len(self._mvn_sensor_setup)
    self._num_joints = len(self._mvn_joint_setup)
    self._num_fingers = None

    self._is_euler = is_euler
    self._is_quaternion = is_quaternion
    self._is_joint_angles = is_joint_angles
    self._is_linear_segments = is_linear_segments
    self._is_angular_segments = is_angular_segments
    self._is_motion_trackers = is_motion_trackers
    self._is_com = is_com
    self._is_time_code = is_time_code

    # Specify the Xsens streaming configuration.
    self._mvn_protocol = mvn_protocol
    self._mvn_ip = mvn_ip
    self._mvn_port = mvn_port

    # Note that the buffer read size must be large enough to receive a full Xsens message.
    #   The longest message is currently from the stream "Position + Orientation (Quaternion)",
    #     which has a message length of 2040 when finger data is enabled.
    self._buffer_read_size = buffer_read_size
    self._buffer_max_size = self._buffer_read_size * 16

    # Initialize state.
    self._buffer: bytes = b''
    self._socket: socket.socket
    self._xsens_sample_index = None # The current Xsens timestep being processed (each timestep will send multiple messages).
    self._xsens_message_start_time_s = None    # When an Xsens message was first received.
    self._xsens_timestep_receive_time_s = None # When the first Xsens message for an Xsens timestep was received.

    stream_info = {
      "mvn_setup": mvn_setup,
      "sampling_rate_hz": sampling_rate_hz,
      "is_euler": is_euler,
      "is_quaternion": is_quaternion,
      "is_joint_angles": is_joint_angles,
      "is_linear_segments": is_linear_segments,
      "is_angular_segments": is_angular_segments,
      "is_motion_trackers": is_motion_trackers,
      "is_com": is_com,
      "is_time_code": is_time_code,
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
  def create_stream(cls, stream_info: dict) -> MvnAnalyzeStream:
    return MvnAnalyzeStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  # Connect to the Xsens network streams, and determine what streams are active.
  def _connect(self) -> bool:
    # Open a socket to the Xsens network stream.
    if MvnNetProto.UDP == self._mvn_protocol:
      self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      self._socket.settimeout(10) # timeout for all socket operations, such as receiving if the Xsens network stream is inactive.
    else:
      raise ValueError("[MVN Analyze Streamer]: currently unsupported network protocol ", self._mvn_protocol)
    return True


  def _keep_samples(self) -> None:
    # Bind the socket after nodes synced, ensures no buffering on the socket happens. 
    self._socket.bind((self._mvn_ip, int(self._mvn_port)))


  # Parse an Xsens message to extract its data.
  def _process_data(self) -> None:
    if self._is_continue_capture:
      self._read_socket_into_buffer()

      message_end_index = self._process_buffer()
      # If the message was complete, remove it from the buffer
      #  and note that we're waiting for a new start code.
      if message_end_index is not None:
        self._buffer = self._buffer[message_end_index+1:]
        self._xsens_message_start_time_s = None
    else:
      self._send_end_packet()


  # Helper to read a specified number of bytes starting at a specified index
  #   and to return an updated index counter.
  def _read_bytes(self, message: bytes, starting_index: int | None, num_bytes: int) -> tuple[bytes | None, int | None]:
    if message is None or starting_index is None or num_bytes is None:
      res = None
      next_index = None
    elif starting_index + num_bytes <= len(message):
      res = message[starting_index : (starting_index + num_bytes)]
      next_index = starting_index + num_bytes
    else:
      res = None
      next_index = None
    return res, next_index


  def _read_socket_into_buffer(self) -> None:
    if MvnNetProto.UDP == self._mvn_protocol:
      try:
        data = self._socket.recv(self._buffer_read_size)
        if len(self._buffer)+len(data) <= self._buffer_max_size:
          self._buffer += data
        else:
          # Remove old data if the buffer is overflowing.
          self._buffer = data
      except socket.timeout:
        print('MVN UDP receive socket timed out on receive.', flush=True)
    else:
      raise ValueError("[MVN Analyze Streamer]: currently unsupported network protocol ", self._mvn_protocol)

    # Record this as the message arrival time if it's the first time
    #  seeing this message start code in the buffer.
    message_start_index = self._buffer.find(MVN_UDP_ID_STRING)
    if message_start_index >= 0 and self._xsens_message_start_time_s is None:
      self._xsens_message_start_time_s = get_time()


  def _process_buffer(self) -> int | None:
    message: bytes = self._buffer

    # Use the starting code to determine the initial index.
    next_index = message.find(MVN_UDP_ID_STRING)
    if next_index < 0:
      return None

    # Try to process the message.
    metadata, next_index = self._process_header(message, next_index)

    # Check that all were present by checking the last one.
    if metadata['payload_size'] is None:
      print('Incomplete MVN message.', flush=True)
      return None

    # Read the payload, and check that it is fully present
    payload, payload_end_index = self._read_bytes(message, next_index, metadata['payload_size'])
    if payload is None:
      print('No MVN message payload yet.', flush=True)
      return None

    extra_data = {
      'counter': metadata['counter'],
      'time_since_start_s': metadata['time_since_start_s'], # from metadata
    }

    data: NewDataDict = {}

    if MvnMsgType.POSE_EULER == metadata['message_type'] or MvnMsgType.POSE_QUATERNION == metadata['message_type']:
      data['xsens-pose'], next_index = self._process_pose(message, next_index, metadata, extra_data)
    elif MvnMsgType.JOINT_ANGLES == metadata['message_type']:
      data['xsens-joints'], next_index = self._process_joint_angles(message, next_index, metadata, extra_data)
    elif MvnMsgType.CENTER_OF_MASS == metadata['message_type']:
      data['xsens-com'], next_index = self._process_com(message, next_index, metadata, extra_data)
    elif MvnMsgType.LINEAR_SEGMENTS == metadata['message_type']:
      data['xsens-linear-segments'], next_index = self._process_linear_segments(message, next_index, metadata, extra_data)
    elif MvnMsgType.ANGULAR_SEGMENTS == metadata['message_type']:
      data['xsens-angular-segments'], next_index = self._process_angular_segments(message, next_index, metadata, extra_data)
    elif MvnMsgType.MOTION_TRACKERS == metadata['message_type']:
      data['xsens-motion-trackers'], next_index = self._process_motion_trackers(message, next_index, metadata, extra_data)
    elif MvnMsgType.TIME_CODE == metadata['message_type']:
      data['xsens-time'], next_index = self._process_time_code(message, next_index, metadata, extra_data)
    else:
      # The message had a type that is not currently being processed/recorded.
      # No processing is required, but the pointer should still be advanced to ignore the message.
      print('Unknown MVN message type: ', metadata['message_type'], flush=True)
      return payload_end_index-1 if payload_end_index is not None else None

    # Check that the entire message was parsed.
    if payload_end_index != next_index and next_index is not None:
      print('The Xsens payload should end at byte %d, but the last byte processed was %d' % (payload_end_index, next_index-1))
      return None

    tag: str = "%s.data" % (self._log_source_tag())
    self._publish(tag=tag, process_time_s=self._xsens_timestep_receive_time_s, data=data)

    # The message was successfully parsed.
    # Return the last index of the message that was used.
    return next_index-1 if next_index is not None else None


  def _process_header(self, message: bytes, next_index: int | None) -> tuple[MvnMetadata, int | None]:
    # Parse the header.
    # Each datagram starts with 24-byte header.
    # NOTE: network byte-order (big-endian).
    start_code,       next_index = self._read_bytes(message, next_index, len(MVN_UDP_ID_STRING))
    message_type,     next_index = self._read_bytes(message, next_index, 2)
    counter,          next_index = self._read_bytes(message, next_index, 4) # message index basically
    datagram_counter, next_index = self._read_bytes(message, next_index, 1) # index of datagram chunk when message sent in multiple datagrams
    num_items,        next_index = self._read_bytes(message, next_index, 1) # number of segments or points in the datagram
    time_code,        next_index = self._read_bytes(message, next_index, 4) # ms since the start of recording
    char_id,          next_index = self._read_bytes(message, next_index, 1) # id of the tracker person (if multiple)
    num_segments,     next_index = self._read_bytes(message, next_index, 1) # we always have 23 body segments
    num_props,        next_index = self._read_bytes(message, next_index, 1) # number of props (swords etc)
    num_fingers,      next_index = self._read_bytes(message, next_index, 1) # number of finger track segments
    reserved,         next_index = self._read_bytes(message, next_index, 2)
    payload_size,     next_index = self._read_bytes(message, next_index, 2) # number of bytes in the actual data of the message

    # Decode (and verify) header information.
    message_type = int(message_type) if message_type is not None else None

    (payload_size,
     counter,
     datagram_counter,
     time_code,
     char_id,
     num_items,
     num_segments,
     num_fingers,
     num_props,
     reserved) = map(lambda val: int.from_bytes(val, byteorder='big', signed=False) if val is not None else None, 
                     [payload_size,
                      counter,
                      datagram_counter,
                      time_code,
                      char_id,
                      num_items,
                      num_segments,
                      num_fingers,
                      num_props,
                      reserved])

    # TODO: enable multi-datagram case. All packets fit in a single datagram size.
    if datagram_counter != (1 << 7):
      print('Not a single last message')
      return MvnMetadataEmpty(), None
    if char_id != 0:
      print('We only support a single person (a single character).')
      return MvnMetadataEmpty(), None

    # If the message type and sample counter are known,
    #  use this time as the best-guess timestamp for the data.
    if message_type is not None and counter is not None:
      if self._xsens_sample_index != counter:
        self._xsens_timestep_receive_time_s = self._xsens_message_start_time_s
        self._xsens_sample_index = counter

    # Organize the metadata so far.
    return MvnMetadata(
      message_type=message_type,
      counter=counter,
      datagram_counter=datagram_counter,
      num_items=num_items,
      time_since_start_s=(time_code*1000) if time_code is not None else None,
      char_id=char_id,
      num_segments=num_segments,
      num_props=num_props,
      num_fingers=num_fingers,
      reserved=reserved,
      payload_size=payload_size), next_index


  def _process_pose(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Mutually-exclusive Euler or Quaternion data contain:
    #   Euler:
    #     Segment positions (x/y/z) in cm
    #     Segment rotation (x/y/z) using a Y-Up and right-handed coordinate system
    #   Quaternion data contains:
    #     Segment positions (x/y/z) in cm
    #     Segment rotation (re/i/j/k) using a Z-Up and right-handed coordinate system
    is_euler = metadata['message_type'] == MvnMsgType.POSE_EULER
    is_quaternion = metadata['message_type'] == MvnMsgType.POSE_QUATERNION
    if is_euler:
      num_orientation_elements = 3
      orientation_stream_name = 'euler'
    else:
      num_orientation_elements = 4
      orientation_stream_name = 'quaternion'

    positions = np.zeros((self._num_segments, 3), dtype=np.float32)
    orientations = np.zeros((self._num_segments, num_orientation_elements), dtype=np.float32)

    # Read the position and rotation of each segment.
    if metadata['num_segments'] is not None:
      for _ in range(metadata['num_segments']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        position, next_index = self._read_bytes(message, next_index, 3*4)
        orientation, next_index = self._read_bytes(message, next_index, num_orientation_elements*4)

        if segment_id is not None and position is not None and orientation is not None:
          segment_id = int.from_bytes(segment_id, byteorder='big', signed=False)
          position = np.array(struct.unpack('!3f', position), np.float32)
          orientation = np.array(struct.unpack('!%df' % num_orientation_elements, orientation), np.float32)

          if segment_id in self._segment_id_mapping.keys():
            # If using the Euler stream, received data is YZX so swap order to get XYZ.
            #   NOTE: that positions from the quaternion stream are already XYZ.
            if is_euler:
              position = position[[2,0,1]]
              orientation = orientation[[2,0,1]]

            # NOTE: segment IDs from Xsens are 1-based,
            #       but otherwise should be usable as the matrix index.
            id = self._segment_id_mapping[segment_id]
            positions[id, :] = position
            orientations[id, :] = orientation

    # NOTE: loops over any prop and glove data because we are not processing it yet.
    if metadata['num_props'] is not None and metadata['num_fingers'] is not None:
      for _ in range(metadata['num_props'] + metadata['num_fingers']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        position, next_index = self._read_bytes(message, next_index, 3*4)
        orientation, next_index = self._read_bytes(message, next_index, num_orientation_elements*4)

    return {
      orientation_stream_name: orientations,
      'position': positions,
      **extra_data
    }, next_index


  def _process_joint_angles(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Joint angle data contains:
    #   The parent and child segments of the joint.
    #     These are represented as a single integer: 256*segment_id + point_id
    #   Rotation around the segment's x/y/z axes in degrees.
    joint_angles = np.zeros((self._num_joints, 3), dtype=np.float32)

    # Read the ids and rotations of each joint.
    if metadata['num_items'] is not None:
      for _ in range(metadata['num_items']):
        point_id_parent, next_index = self._read_bytes(message, next_index, 4)
        point_id_child, next_index = self._read_bytes(message, next_index, 4)
        angle_over_segment, next_index = self._read_bytes(message, next_index, 3*4)

        if point_id_parent is not None and point_id_child is not None and angle_over_segment is not None:
          point_id_parent = int.from_bytes(point_id_parent, byteorder='big', signed=False)
          point_id_child = int.from_bytes(point_id_child, byteorder='big', signed=False)
          angle_over_segment = np.array(struct.unpack('!3f', angle_over_segment), np.float32)

          # Convert these id's into mapping into the numpy array.
          joint_id = point_id_parent << 16 + point_id_child

          if joint_id in self._joint_id_mapping.keys():
            joint_angles[self._joint_id_mapping[joint_id], :] = angle_over_segment

    return {
      'angle': joint_angles,
      **extra_data
    }, next_index


  def _process_com(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Center of mass data contains:
    #   x/y/z position in cm.
    position, next_index = self._read_bytes(message, next_index, 3*4)
    velocity, next_index = self._read_bytes(message, next_index, 3*4)
    acceleration, next_index = self._read_bytes(message, next_index, 3*4)

    if position is not None and velocity is not None and acceleration is not None:
      position = np.array(struct.unpack('!3f', position), np.float32)
      velocity = np.array(struct.unpack('!3f', velocity), np.float32)
      acceleration = np.array(struct.unpack('!3f', acceleration), np.float32)

    return {
      'position': position,
      'velocity': velocity,
      'acceleration': acceleration,
      **extra_data
    }, next_index


  def _process_linear_segments(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Linear segments data contains:
    #   The global position, velocity and acceleration of the segment.
    positions = np.zeros((self._num_segments, 3), dtype=np.float32)
    velocities = np.zeros((self._num_segments, 3), dtype=np.float32)
    accelerations = np.zeros((self._num_segments, 3), dtype=np.float32)

    if metadata['num_items'] is not None:
      for _ in range(metadata['num_items']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        position, next_index = self._read_bytes(message, next_index, 3*4)
        velocity, next_index = self._read_bytes(message, next_index, 3*4)
        acceleration, next_index = self._read_bytes(message, next_index, 3*4)

        if segment_id is not None and position is not None and velocity is not None and acceleration is not None:
          segment_id = int.from_bytes(segment_id, byteorder='big', signed=False)
          position = np.array(struct.unpack('!3f', position), np.float32)
          velocity = np.array(struct.unpack('!3f', velocity), np.float32)
          acceleration = np.array(struct.unpack('!3f', acceleration), np.float32)

          if segment_id in self._segment_id_mapping.keys():
            id = self._segment_id_mapping[segment_id]
            positions[id, :] = position
            velocities[id, :] = velocity
            accelerations[id, :] = acceleration

    # NOTE: loops over any prop and glove data because we are not processing it yet.
    if metadata['num_props'] is not None and metadata['num_fingers'] is not None:
      for _ in range(metadata['num_props'] + metadata['num_fingers']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        position, next_index = self._read_bytes(message, next_index, 3*4)
        velocity, next_index = self._read_bytes(message, next_index, 3*4)
        acceleration, next_index = self._read_bytes(message, next_index, 3*4)

    return {
      'position': positions,
      'velocity': velocities,
      'acceleration': accelerations,
      **extra_data
    }, next_index


  def _process_angular_segments(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Angular segments data contains:
    #   The quaternion orientation, angular velocity and angular acceleration of the segment.
    orientations = np.zeros((self._num_segments, 4), dtype=np.float32)
    velocities = np.zeros((self._num_segments, 3), dtype=np.float32)
    accelerations = np.zeros((self._num_segments, 3), dtype=np.float32)

    if metadata['num_items'] is not None:
      for _ in range(metadata['num_items']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        orientation, next_index = self._read_bytes(message, next_index, 4*4)
        velocity, next_index = self._read_bytes(message, next_index, 3*4)
        acceleration, next_index = self._read_bytes(message, next_index, 3*4)

        if (segment_id is not None and
            orientation is not None and
            velocity is not None and
            acceleration is not None):
          segment_id = int.from_bytes(segment_id, byteorder='big', signed=False)
          orientation = np.array(struct.unpack('!4f', orientation), np.float32)
          velocity = np.array(struct.unpack('!3f', velocity), np.float32)
          acceleration = np.array(struct.unpack('!3f', acceleration), np.float32)

          if segment_id in self._segment_id_mapping.keys():
            id = self._segment_id_mapping[segment_id]
            orientations[id, :] = orientation
            velocities[id, :] = velocity
            accelerations[id, :] = acceleration

    # NOTE: loops over any prop and glove data because we are not processing it yet.
    if metadata['num_props'] is not None and metadata['num_fingers'] is not None:
      for _ in range(metadata['num_props'] + metadata['num_fingers']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        orientation, next_index = self._read_bytes(message, next_index, 4*4)
        velocity, next_index = self._read_bytes(message, next_index, 3*4)
        acceleration, next_index = self._read_bytes(message, next_index, 3*4)

    return {
      'quaternion': orientations,
      'velocity': velocities,
      'acceleration': accelerations,
      **extra_data
    }, next_index


  def _process_motion_trackers(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Trackers data contains:
    #   The quaternion orientation for the segment with a tracker, free acceleration and magnetic field.
    orientations = np.zeros((self._num_sensors, 4), dtype=np.float32)
    global_free_accelerations = np.zeros((self._num_sensors, 3), dtype=np.float32)
    local_accelerations = np.zeros((self._num_sensors, 3), dtype=np.float32)
    local_gyroscopes = np.zeros((self._num_sensors, 3), dtype=np.float32)
    local_magnetometers = np.zeros((self._num_sensors, 3), dtype=np.float32)

    if metadata['num_items'] is not None:
      for _ in range(metadata['num_items']):
        segment_id, next_index = self._read_bytes(message, next_index, 4)
        orientation, next_index = self._read_bytes(message, next_index, 4*4)
        global_free_acceleration, next_index = self._read_bytes(message, next_index, 3*4)
        local_acceleration, next_index = self._read_bytes(message, next_index, 3*4)
        local_gyroscope, next_index = self._read_bytes(message, next_index, 3*4)
        local_magnetometer, next_index = self._read_bytes(message, next_index, 3*4)

        if (segment_id is not None and
            orientation is not None and
            global_free_acceleration is not None and
            local_acceleration is not None and
            local_gyroscope is not None and
            local_magnetometer is not None):
          segment_id = int.from_bytes(segment_id, byteorder='big', signed=False)
          orientation = np.array(struct.unpack('!4f', orientation), np.float32)
          global_free_acceleration = np.array(struct.unpack('!3f', global_free_acceleration), np.float32)
          local_acceleration = np.array(struct.unpack('!3f', local_acceleration), np.float32)
          local_gyroscope = np.array(struct.unpack('!3f', local_gyroscope), np.float32)
          local_magnetometer = np.array(struct.unpack('!3f', local_magnetometer), np.float32)

          if segment_id in self._sensor_id_mapping.keys():
            id = self._sensor_id_mapping[segment_id]
            orientations[id, :] = orientation
            global_free_accelerations[id, :] = global_free_acceleration
            local_accelerations[id, :] = local_acceleration
            local_gyroscopes[id, :] = local_gyroscope
            local_magnetometers[id, :] = local_magnetometer

    return {
      'quaternion': orientations,
      'free_acceleration': global_free_accelerations,
      'acceleration': local_accelerations,
      'gyroscope': local_gyroscopes,
      'magnetometer': local_magnetometers,
      **extra_data
    }, next_index


  def _process_time_code(self, message: bytes, next_index: int | None, metadata: MvnMetadata, extra_data: MvnCommonData) -> MvnDataTuple:
    # Time data contains:
    #  A string for the sample timestamp formatted as HH:MM:SS.mmm
    str_length, next_index = self._read_bytes(message, next_index, 4)
    time_code_s = float('nan')
    if str_length is not None:
      str_length = int.from_bytes(str_length, byteorder='big', signed=True)
      
      if str_length != 12:
        print('Unexpected number of bytes in the time code string: %d instead of 12' % str_length)

      time_code_str, next_index = self._read_bytes(message, next_index, str_length)
      if time_code_str is not None:  
        time_code_str = time_code_str.decode('utf-8')
        # The received string is a time in UTC without a date.
        # Convert this to local time with the current date, then to seconds since epoch.
        time_code_s = get_time_s_from_utc_time_no_date_str(time_code_str, input_time_format='%H:%M:%S.%f')

    return {
      'timestamp_s': time_code_s,
      **extra_data
    }, next_index


  def _stop_new_data(self) -> None:
    self._socket.close()


  def _cleanup(self) -> None:
    super()._cleanup()
