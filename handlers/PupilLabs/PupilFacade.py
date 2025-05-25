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

import queue
import threading
from typing import Any, Callable, Mapping

import msgpack
import numpy as np
import zmq

from utils.sensor_utils import estimate_transmission_delay
from utils.time_utils import get_time


class PupilFacade:
  def __init__(self,
               is_binocular: bool,
               is_stream_video_world: bool,
               is_stream_video_eye: bool,
               is_stream_fixation: bool,
               is_stream_blinks: bool,
               pupil_capture_ip: str,
               pupil_capture_port: str,
               gaze_estimate_stale_s: float,
               video_image_format: str) -> None:
    
    self._is_binocular = is_binocular
    self._is_stream_video_world = is_stream_video_world
    self._is_stream_video_eye = is_stream_video_eye
    self._is_stream_fixation = is_stream_fixation
    self._is_stream_blinks = is_stream_blinks
    self._pupil_capture_ip = pupil_capture_ip
    self._pupil_capture_port = pupil_capture_port
    self._video_image_format = video_image_format
    self._gaze_estimate_stale_s = gaze_estimate_stale_s
    self._is_ipc_capturing = True
    self._start_index_eye = [None] * (2 if is_binocular else 1)
    self._start_index_world = None
    self._previous_index_eye = [0] * (2 if is_binocular else 1)
    self._previous_index_world = 0

    # Connect to the Pupil Capture socket.
    self._zmq_context = zmq.Context.instance()
    self._zmq_requester: zmq.SyncSocket = self._zmq_context.socket(zmq.REQ)
    self._zmq_requester.RCVTIMEO = 2000 # receive timeout in milliseconds
    self._zmq_requester.connect('tcp://%s:%s' % (self._pupil_capture_ip, self._pupil_capture_port))
    # Get the port that will be used for data.
    self._ipc_sub_port = self._send_to_ipc(payload='SUB_PORT')

    # Sync the Pupil Core clock with the system clock.
    self._sync()

    # Subscribe to the desired topics.
    self._topics = ['notify.', 'gaze.3d.%s'%('01.' if self._is_binocular else '0.')]
    if self._is_stream_video_world:
      self._topics.append('frame.world')
    if self._is_stream_video_eye:
      self._topics.append('frame.eye.')
    if self._is_stream_fixation:
      self._topics.append('fixations')
    if self._is_stream_blinks:
      self._topics.append('blinks')

    self._is_more = True
    self._receiver: zmq.SyncSocket = self._zmq_context.socket(zmq.SUB)
    self._receiver.connect('tcp://%s:%s' % (self._pupil_capture_ip, self._ipc_sub_port))
    self._receive_queue: queue.Queue[tuple[list[bytes], float, float]] = queue.Queue()
    self._receive_thread = threading.Thread(target=self._receive_data)


  # Start enqueing packets only after the Nodes synced.
  def keep_data(self) -> None:
    for t in self._topics: self._receiver.subscribe(t)
    self._receive_thread.start()


  def _receive_data(self):
    poller = zmq.Poller()
    poller.register(self._receiver, zmq.POLLIN)
    is_continue = True
    while is_continue:
      poll_res = poller.poll(5)
      if len(poll_res):
        data = self._receiver.recv_multipart()
        time_s = get_time()
        device_time_s = self._get_device_time()
        self._receive_queue.put((data, time_s, device_time_s))
      elif not self._is_more:
        poller.unregister(self._receiver)
        is_continue = False


  # Receive data and return a parsed dictionary.
  def process_data(self) -> tuple[float, Mapping[str, Mapping[str, Any]]] | None:
    try:
      data, time_s, device_time_s = self._receive_queue.get(timeout=5)
      topic = data[0].decode('utf-8')
      output: dict[str, dict[str, Any]] = {
        'eye-time': {
          'device_time_s': device_time_s
        }
      }
      # Process gaze/pupil data
      # Note it works for both, mono- and binocular gaze data
      if topic in ['gaze.2d.0.', 'gaze.3d.0.','gaze.2d.01.', 'gaze.3d.01.']: # former two - monocular, latter two - binocular 
        payload: dict[str, Any] = msgpack.loads(data[1]) # type: ignore
        pupil_data = payload['base_data'] # pupil detection on which the gaze detection was based (just use the first one for now if there were multiple)
        # Record data common to both 2D and 3D formats
        gaze_items = [
          ('timestamp'  , payload['timestamp']),  # seconds from an arbitrary reference time, but should be synced with the video timestamps
          ('position'   , payload['norm_pos']),   # normalized units [0-1]
          ('confidence' , payload['confidence']), # gaze confidence [0-1]
        ]
        pupil_items = [
          ('timestamp'  , [pupil['timestamp'] for pupil in pupil_data]),  # seconds from an arbitrary reference time, but should be synced with the video timestamps
          ('position'   , [pupil['norm_pos'] for pupil in pupil_data]),   # normalized units [0-1]
          ('confidence' , [pupil['confidence'] for pupil in pupil_data]), # [0-1]
          ('diameter'   , [pupil['diameter'] for pupil in pupil_data]),   # 2D image space, unit: pixel
        ]
        # Add extra data available for 3D formats
        if topic in ['gaze.3d.0.', 'gaze.3d.01.']:
          gaze_items.extend([
            ('normal_3d' , list(payload['gaze_normal%s_3d' % ('s' if self._is_binocular else '')].values())),    # [(x,y,z),]
            ('point_3d'  , payload['gaze_point_3d']),     # x,y,z
            ('eye_center_3d' , list(payload['eye_center%s_3d' % ('s' if self._is_binocular else '')].values())), # [(x,y,z),]
          ])
          pupil_items.extend([
            ('polar_theta' , [pupil['theta'] for pupil in pupil_data]),
            ('polar_phi'   , [pupil['phi'] for pupil in pupil_data]),
            ('circle3d_radius' , [pupil['circle_3d']['radius'] for pupil in pupil_data]), # mm in 3D space
            ('circle3d_center' , [pupil['circle_3d']['center'] for pupil in pupil_data]), # mm in 3D space
            ('circle3d_normal' , [pupil['circle_3d']['normal'] for pupil in pupil_data]), # mm in 3D space
            ('diameter3d'      , [pupil['diameter_3d'] for pupil in pupil_data]), # mm in 3D space
            ('sphere_center' , [pupil['sphere']['center'] for pupil in pupil_data]), # mm in 3D space
            ('sphere_radius' , [pupil['sphere']['radius'] for pupil in pupil_data]), # mm in 3D space
            ('projected_sphere_center' , [pupil['projected_sphere']['center'] for pupil in pupil_data]), # pixels in image space
            ('projected_sphere_axes'   , [pupil['projected_sphere']['axes'] for pupil in pupil_data]),   # pixels in image space
            ('projected_sphere_angle'  , [pupil['projected_sphere']['angle'] for pupil in pupil_data]),
          ])
        # Add extra data available for 2D formats
        else:
          pupil_items.extend([
            ('ellipse_center'   , [pupil['ellipse']['center'] for pupil in pupil_data]), # pixels, in image space
            ('ellipse_axes'     , [pupil['ellipse']['axes'] for pupil in pupil_data]),   # pixels, in image space
            ('ellipse_angle_deg', [pupil['ellipse']['angle'] for pupil in pupil_data]),  # degrees
          ])
        output['eye-pupil'] = dict(pupil_items)
        output['eye-gaze'] = dict(gaze_items)
      # Process fixations data
      elif topic == 'fixations':
        payload: dict[str, Any] = msgpack.loads(data[1]) # type: ignore
        output['eye-fixations'] = {
          'id':             payload['id'],             # int
          'timestamp':      payload['timestamp'],      # float
          'norm_pos':       payload['norm_pos'],       # float[2]
          'dispersion':     payload['dispersion'],     # float
          'duration':       payload['duration'],       # float
          'confidence':     payload['confidence'],     # float
          'gaze_point_3d':  payload['gaze_point_3d'],  # float[3]
        }
      # Process blinks data
      elif topic == 'blinks': 
        payload: dict[str, Any] = msgpack.loads(data[1]) # type: ignore
        output['eye-blinks'] = {
          'timestamp':  payload['timestamp'],  # float
          'confidence': payload['confidence'], # float
        }
      # Process world video data
      elif topic == 'frame.world':
        # Prepare the metadata for the frame.
        metadata: dict[str, Any] = msgpack.loads(data[1]) # type: ignore
        if self._start_index_world is None: 
          self._start_index_world = metadata['index']
          is_keyframe = True
        else:
          is_keyframe = (metadata['index'] - self._previous_index_world) > 1 # NOTE: not safe against overflow, but uint64
        self._previous_index_world = metadata['index']
        frame_index = metadata['index'] - self._start_index_world
        # Decode the frame.
        frame_buffer = data[2]
        output['eye-video-world'] = {
          'frame_timestamp': float(metadata['timestamp']),
          'frame_index': frame_index, # world view frame index used for annotation
          'frame_sequence_id': metadata['index'],
          'frame': (frame_buffer, is_keyframe, frame_index),
        }
      # Process eye video data
      elif topic in ['frame.eye.0', 'frame.eye.1']:
        # Prepare the metadata for the frame.
        eye_id = int(topic.split('.')[2])
        metadata: dict[str, Any] = msgpack.loads(data[1]) # type: ignore
        if self._start_index_eye[eye_id] is None: 
          self._start_index_eye[eye_id] = metadata['index']
          is_keyframe = True
        else:
          is_keyframe = (metadata['index'] - self._previous_index_eye[eye_id]) > 1 # NOTE: not safe against overflow, but uint64
        self._previous_index_eye[eye_id] = metadata['index']
        frame_index = metadata['index'] - self._start_index_eye[eye_id]
        # Decode the frame.
        frame_buffer = data[2]
        # Prepare the output for the file writer.
        output['eye-video-eye%d'%eye_id] = {
          'frame_timestamp': float(metadata['timestamp']),
          'frame_index': frame_index,
          'frame_sequence_id': metadata['index'],
          'frame': (frame_buffer, is_keyframe, frame_index)
        }
      return time_s, output
    except queue.Empty:
      if self._is_more:
        print('Timeout waiting on new packets from Pupil Core, maybe streamed paused by user.', flush=True)
      else:
        print('No more packets will come from Pupil Core.', flush=True)
      return None


  # Alternate between capturing data so stream can be temporarily paused.
  def toggle_capturing(self) -> bool:
    self._is_ipc_capturing = not self._is_ipc_capturing
    if not self._is_ipc_capturing:
      for t in self._topics: self._receiver.unsubscribe(t)
    else:
      for t in self._topics: self._receiver.subscribe(t)
    return self._is_ipc_capturing


  def set_stream_data_getter(self, fn: Callable) -> None:
    self._get_latest_stream_data_fn = fn


  # Close sockets used by the Facade, destroy the ZeroMQ context in the SensorStreamer
  def close(self) -> None:
    for t in self._topics: self._receiver.unsubscribe(t)
    self._is_more = False
    self._receive_thread.join()
    self._zmq_requester.close()
    self._receiver.close()


  # A helper to clear the Pupil socket receive buffer.
  def _flush_buffer(self, socket: zmq.SyncSocket) -> None:
    flush_completed = False
    while not flush_completed:
      try:
        socket.recv(flags=zmq.NOBLOCK)
      except:
        flush_completed = True


  # A helper method to send data to the pupil system.
  # Payload can be a dict or a simple string.
  # Strings will be sent as-is, while a dict will be sent after a topic message.
  # Returns the response message received.
  def _send_to_ipc(self, payload: dict | str, topic: str | None = None) -> str:
    # Try to receive any outstanding messages, since sending
    #  will fail if there are any waiting.
    self._flush_buffer(socket=self._zmq_requester)
    # Send the desired data as a dict or string.
    if isinstance(payload, dict):
      # Pack and send the payload.
      encoded_payload = msgpack.dumps(payload)
      self._zmq_requester.send_string(topic if topic is not None else 'notify.%s' % payload['subject'], flags=zmq.SNDMORE)
      self._zmq_requester.send(encoded_payload)
    else:
      # Send the topic if there is one.
      if topic is not None:
        self._zmq_requester.send_string(topic, flags=zmq.SNDMORE)
      # Send the payload as a string.
      self._zmq_requester.send_string(payload)
    # Receive the response.
    return self._zmq_requester.recv_string()


  # Get the time of the Pupil Core clock.
  # Data exported from the Pupil Capture software uses timestamps
  #  that are relative to a random epoch time.
  def _get_device_time(self) -> float:
    pupil_time_str = self._send_to_ipc('t')
    return float(pupil_time_str)


  # Set the time of the Pupil Core clock to the system time.
  def _sync(self) -> bool:
    # self._log_status('Syncing the Pupil Core clock with the system clock')
    # Note that the same number of decimals will always be used,
    #  so the length of the message is always the same
    #  (this can help make delay estimates more accurate).
    def set_device_time(time_s: float):
      self._send_to_ipc('T %0.8f' % time_s)

    # Estimate the network delay when sending the set-time command.
    transmit_delay_s = estimate_transmission_delay(ping_fn=lambda: set_device_time(get_time()))

    # self._log_debug('Estimated Pupil Core set clock transmit delay [ms]: mean %0.3f | std %0.3f | min %0.3f | max %0.3f' % \
    #                 (np.mean(transmit_delay_s)*1000.0, np.std(transmit_delay_s)*1000.0,
    #                  np.min(transmit_delay_s)*1000.0, np.max(transmit_delay_s)*1000.0))
    # Check that the sync was successful.
    set_device_time(get_time() + transmit_delay_s)
    clock_offset_ms = self._get_device_clock_offset_s()/1000.0
    if abs(clock_offset_ms) > 5:
      # self._log_warn('WARNING: Pupil Core clock sync may not have been successful. Offset is still %0.3f ms.' % clock_offset_ms)
      return False
    return True


  # Measure the offset between the Pupil Core clock (relative to a random epoch)
  #  and the system clock (relative to the standard epoch).
  # See the following for more information:
  #  https://docs.pupil-labs.com/core/terminology/#timestamps
  #  https://github.com/pupil-labs/pupil-helpers/blob/6e2cd2fc28c8aa954bfba068441dfb582846f773/python/simple_realtime_time_sync.py#L119
  def _get_device_clock_offset_s(self, num_samples: int = 100) -> float:
    assert num_samples > 0, 'Measuring the Pupil Capture clock offset requires at least one sample'
    clock_offsets_s = []
    for i in range(num_samples):
      # Account for network delays by recording the local time
      #  before and after the call to fetch the pupil time
      #  and then assuming that the clock was measured at the midpoint
      #  (assume symmetric network delays).
      # Note that in practice, this delay is small when
      #  using Pupil Capture via USB (typically 0-1 ms, rarely 5-10 ms).
      local_time_before = get_time()
      pupil_time = self._get_device_time()
      local_time_after = get_time()
      local_time = (local_time_before + local_time_after) / 2.0
      clock_offsets_s.append(pupil_time - local_time)
    # Average multiple readings to account for variable network delays.
    # self._log_debug('Estimated Pupil Core clock offset [ms]: mean %0.3f | std %0.3f | min %0.3f | max %0.3f' % \
    #                 (np.mean(clock_offsets_s)*1000.0, np.std(clock_offsets_s)*1000.0,
    #                  np.min(clock_offsets_s)*1000.0, np.max(clock_offsets_s)*1000.0))
    # TODO: remove outliers before averaging
    res = np.mean(clock_offsets_s)
    return float(res)

  # TODO: periodically sync the glasses time?
