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

import copy
from collections import OrderedDict, deque
from typing import Any, Iterable, Iterator, Mapping
import cv2
import dash_bootstrap_components as dbc
from threading import Lock

from utils.dict_utils import convert_dict_values_to_str
from utils.time_utils import get_time
from utils.types import VIDEO_FORMAT, DataFifoDict, DeviceLockDict, ExtraDataInfoDict, NewDataDict, StreamInfoDict



#########################################################################
#########################################################################
# An abstract class to store data of a Producer.
#   May contain multiple streams from the sensor, 
#     such as acceleration and gyroscope from the DOTs.
# Structure of data and streams_info:
#   Dict with device names as keys, each of which maps to:
#     Dict with name of streams, each of which maps to:
#       for data: 'data', 'time_s', and others if desired
#       for streams_info: 'data_type', 'sample_size', 'sampling_rate_hz',
#         'timesteps_before_solidified', 'extra_data_info'
# Can periodically clear old data (if needed).
#########################################################################
#########################################################################
class Stream(ABC):
  # Will store the class name of each sensor in HDF5 metadata,
  #   to facilitate recreating classes when replaying the logs later.
  # The following is the metadata key to store that information.
  metadata_class_name_key = 'SensorStreamer class name'
  # Will look for a special metadata key that labels data channels,
  #   to use for logging purposes and general user information.
  metadata_data_headings_key = 'Data headings'

  _metadata: Mapping
  _data: DataFifoDict
  _streams_info: StreamInfoDict
  _locks: DeviceLockDict

  def __init__(self) -> None:
    self._metadata = dict()
    self._data = dict()
    self._streams_info = dict()
    # NOTE: Lock used only to delegate access to the start of the FIFO
    #   in case a Consumer is interested in only some freshest data elements.
    #   This allows the end of the FIFO to be saved and discarded by the Logger.
    self._locks = dict()


  ############################
  ###### INTERFACE FLOW ######
  ############################
  # Get actual frame rate, subject to expected transmission delay and throughput limitation,
  #   to confidently judge the performance of the system.
  # Computed based on how fast data becomes available to the data structure, hence suitable
  #   to measure frame rate on the subscriber, local or remote.
  @abstractmethod
  def get_fps(self) -> dict[str, float]:
    pass


  # Get how each stream should be visualized.
  # Should be overridden by subclasses if desired.
  # Returns a Dash `Row`.
  @abstractmethod
  def build_visulizer(self) -> dbc.Row | None:
    return None


  #############################
  ###### GETTERS/SETTERS ######
  #############################
  # Add a new device stream.
  # Will by default add a stream for each device to mark host time of the arrival of a sample into the foreground thread.
  # @param timesteps_before_solidified allows indication that data/timestamps for
  #   previous timesteps may be altered when a new sample is received.
  #   For example, some sensors may send two samples at the same time, and the
  #     streamer class may infer a timestamp for the previous reading when
  #     a new pair of readings arrive.  A timestamp is thus not 'final' until the next timestep.
  # @param extra_data_info is used to specify additional data keys that will be streamed
  #   along with the default 'data'.
  #   It should be a dict, where each extra data key maps to a dict with at least 'data_type' and 'sample_size'.
  # @param data_notes can be a string or a dict of relevant info.
  #   If it's a dict, the key 'Data headings' is recommended and will be used by DataLogger for headers.
  #     In that case, 'Data headings' should map to a list of strings of length sample_size.
  def add_stream(self,
                 device_name: str,
                 stream_name: str,
                 data_type: str,
                 sample_size: Iterable[int],
                 sampling_rate_hz: float,
                 is_measure_rate_hz: bool = False,
                 data_notes: str | dict = {},
                 is_video: bool = False,
                 color_format: str = None,
                 is_audio: bool = False,
                 timesteps_before_solidified: int = 0,
                 extra_data_info: ExtraDataInfoDict = {}) -> None:
    if stream_name == 'process_time_s':
      raise ValueError('\'process_time_s\' is reserved for Stream internal use.')
    self._add_stream(device_name=device_name,
                     stream_name=stream_name,
                     data_type=data_type,
                     sample_size=sample_size,
                     sampling_rate_hz=sampling_rate_hz,
                     is_measure_rate_hz=is_measure_rate_hz,
                     data_notes=data_notes,
                     is_video=is_video,
                     color_format=color_format,
                     is_audio=is_audio,
                     timesteps_before_solidified=timesteps_before_solidified,
                     extra_data_info=extra_data_info)
    if 'process_time_s' not in self._data[device_name]:
      self._add_stream(device_name=device_name,
                       stream_name='process_time_s',
                       data_type='float64',
                       sample_size=(1,),
                       sampling_rate_hz=0.0,
                       data_notes='Time of arrival of the data point to the host PC, ' \
                                  'to be used for aligned idexing of data between distributed hosts.')


  def _add_stream(self,
                  device_name: str,
                  stream_name: str,
                  data_type: str,
                  sample_size: Iterable[int],
                  sampling_rate_hz: float,
                  is_measure_rate_hz: bool = False,
                  data_notes: str | dict = {},
                  is_video: bool = False,
                  color_format: str = None,
                  is_audio: bool = False,
                  timesteps_before_solidified: int = 0,
                  extra_data_info: ExtraDataInfoDict = {}) -> None:
    self._locks.setdefault(device_name, Lock())
    self._streams_info.setdefault(device_name, dict())
    if not isinstance(sample_size, Iterable):
      sample_size = [sample_size]
    self._streams_info[device_name][stream_name] = dict([
      ('data_type', data_type),
      ('sample_size', sample_size),
      ('data_notes', data_notes),
      ('sampling_rate_hz', sampling_rate_hz),
      ('is_measure_rate_hz', is_measure_rate_hz),
      ('is_video', is_video),
      ('is_audio', is_audio),
      ('timesteps_before_solidified', timesteps_before_solidified),
      ('extra_data_info', extra_data_info)])
    # Record color formats to use by FFmpeg and OpenCV, for saving and displaying frames.
    if is_video:
      try:
        video_format = VIDEO_FORMAT[color_format]
        self._streams_info[device_name][stream_name]['ffmpeg_input_format'] = video_format.ffmpeg_input_format
        self._streams_info[device_name][stream_name]['color_format'] = {'ffmpeg': video_format.ffmpeg_pix_fmt, 'cv2': video_format.cv2_cvt_color}
      except KeyError:
        print("Color format %s is not supported when specifying video frame pixel color format on Stream."%color_format)
    # Some metadata to keep track of during running to measure the actual frame rate.
    if is_measure_rate_hz:
      # Set at start actual rate equal to desired sample rate
      self._streams_info[device_name][stream_name]['actual_rate_hz'] = self._streams_info[device_name][stream_name]['sampling_rate_hz']
      # Create a circular buffer of 1 second, w.r.t. desired sample rate
      circular_buffer_len: int = max(round(sampling_rate_hz), 1)
      self._streams_info[device_name][stream_name]['dt_circular_buffer'] = list([1/sampling_rate_hz] * circular_buffer_len)
      self._streams_info[device_name][stream_name]['dt_circular_index'] = 0
      self._streams_info[device_name][stream_name]['dt_running_sum'] = 1.0
      self._streams_info[device_name][stream_name]['old_toa'] = get_time()
    self.clear_data(device_name, stream_name)


  # Appending data to the Deque is thread-safe,
  #   but need to lock so reverse iterator doesn't throw immutability error
  #   (i.e. when GUI gets the newest N samples, while Node appends new data and Logger pops the oldest).
  def append_data(self, process_time_s: float, data: NewDataDict) -> None:
    for (device_name, streams_data) in data.items():
      if streams_data is not None:
        self._locks[device_name].acquire()
        for (stream_name, stream_data) in streams_data.items():
          self._append(device_name, stream_name, stream_data)
        self._append(device_name, 'process_time_s', process_time_s)
        self._locks[device_name].release()


  # Add a single timestep of data to the data log.
  # @param time_s and @param data should each be a single value.
  # @param extra_data should be a dict mapping each extra data key to a single value.
  #   The extra data keys should match what was specified in add_stream() or set_streams_info().
  def _append(self,
              device_name: str,
              stream_name: str,
              data: Any) -> None:
    self._data[device_name][stream_name].append(data)

    # If stream set to measure actual fps
    # TODO: cleanup to use a fixed-length Deque instead.
    if self._streams_info[device_name][stream_name]['is_measure_rate_hz']:
      # Make intermediate variables for current and previous samples' time-of-arrival
      new_toa = get_time()
      old_toa = self._streams_info[device_name][stream_name]['old_toa']
      # Record the new arrival time for the next iteration
      self._streams_info[device_name][stream_name]['old_toa'] = new_toa
      # Update the running sum of time increments of the circular buffer  
      oldest_dt = self._streams_info[device_name][stream_name]['dt_circular_buffer']\
                    [self._streams_info[device_name][stream_name]['dt_circular_index']]
      newest_dt = new_toa - old_toa
      self._streams_info[device_name][stream_name]['dt_running_sum'] += (newest_dt - oldest_dt)
      # Put current time increment in place of the oldest one in the circular buffer
      self._streams_info[device_name][stream_name]['dt_circular_buffer']\
        [self._streams_info[device_name][stream_name]['dt_circular_index']] = newest_dt
      # Move the index in the circular fashion
      self._streams_info[device_name][stream_name]['dt_circular_index'] =\
        (self._streams_info[device_name][stream_name]['dt_circular_index'] + 1) %\
        len(self._streams_info[device_name][stream_name]['dt_circular_buffer'])
      # Refresh the actual frame rate information
      self._streams_info[device_name][stream_name]['actual_rate_hz'] = \
        len(self._streams_info[device_name][stream_name]['dt_circular_buffer']) / \
        self._streams_info[device_name][stream_name]['dt_running_sum']


  # Pop FIFO data starting with the first timestep that hasn't been logged yet,
  #   and ending at the most recent data (or back by a few timesteps
  #   if the streamer may still edit the most recent timesteps).
  # NOTE: Deque popping is thread-safe while appending.
  # Cleans up the oldest data in the FIFO, so can be called only once. 
  #   for x in pop_data(..., num_to_pop):
  # Returns an iterator over the same data elements placed using 'append'.
  # Passing an iterator to either method, will consume all available data in that stream,
  #   so if HDF5 and CSV requested from the same stream, only HDF5 will be written.
  #   This will effectively clear logged data from memory.
  def pop_data(self, 
               device_name: str, 
               stream_name: str,
               num_oldest_to_pop: int = None,
               is_flush: bool = False) -> Iterator[Any]:
    # O(1) complexity to check length of a Deque.
    num_available: int = len(self._data[device_name][stream_name])
    # Can pop all available data, except what must be kept peekable.
    num_poppable: int = num_available - self._streams_info[device_name][stream_name]['timesteps_before_solidified']
    # If experiment ended, flush all available data from the Stream.
    if is_flush:
      num_oldest_to_pop = num_available
    elif num_oldest_to_pop is None:
      num_oldest_to_pop = num_poppable
    else:
      num_oldest_to_pop = min(num_oldest_to_pop, num_poppable)
    # Iterate through the doubly-linked list, clearing popped data, while new data is added to it.
    num_popped: int = 0
    while num_popped < num_oldest_to_pop:
      yield self._data[device_name][stream_name].popleft()
      num_popped += 1


  # Look at the N newest data elements.
  #   Locks the Stream from appends, but permits popping from the other end.
  #   (i.e. GUI temporarily locks new appends to visualize N latest samples, while Logger flushes the other end).
  # Returns an iterator.
  def peek_data_new(self,
                    device_name: str,
                    stream_name: str,
                    num_newest_to_peek: int = None) -> Iterator[Any]:
    self._locks[device_name].acquire()
    num_peekable: int = min(self._streams_info[device_name][stream_name]['timesteps_before_solidified'], 
                            len(self._data[device_name][stream_name]))
    if num_newest_to_peek is None:
      num_newest_to_peek = num_peekable
    else:
      num_newest_to_peek = min(num_newest_to_peek, 
                               self._streams_info[device_name][stream_name]['timesteps_before_solidified'])
    num_peeked: int = 0
    # Get an iterator to traverse the linked list from the write end (newest data) -> O(1).
    stream_reversed = reversed(self._data[device_name][stream_name])
    while num_peeked < num_newest_to_peek:
      yield next(stream_reversed)
      num_peeked += 1
    self._locks[device_name].release()


  # Clear data for a stream (and add the stream if it doesn't exist).
  # Optionally, can clear the oldest N data elements.
  def clear_data(self,
                 device_name: str,
                 stream_name: str,
                 num_oldest_to_clear: int = None) -> None:
    # Create the device/stream entry if it doesn't exist, else clear it.
    self._data.setdefault(device_name, OrderedDict())
    if stream_name not in self._data[device_name]:
      self._data[device_name][stream_name] = deque()
    elif num_oldest_to_clear is not None:
      # Clearing up to a point in the Deque.
      # Wait until neither Node, nor GUI, append or peek newest data, respectively,
      #   only if clearing past their operating area.
      num_clearable: int = len(self._data[device_name][stream_name]) - self._streams_info[device_name][stream_name]['timesteps_before_solidified']
      is_to_lock: bool = not (num_oldest_to_clear < num_clearable)
      num_cleared: int = 0
      if is_to_lock: self._locks[device_name].acquire()
      while num_cleared < num_oldest_to_clear:
        self._data[device_name][stream_name].popleft()
        num_cleared += 1
      if is_to_lock: self._locks[device_name].release()
    else:
      # Clearing the whole Deque. 
      # Wait until neither Node, nor GUI, append or peek newest data, respectively. 
      self._locks[device_name].acquire()
      self._data[device_name][stream_name].clear()
      self._locks[device_name].release()


  # Clear all streams of all devices in the Stram datastructure.
  def clear_data_all(self) -> None:
    for (device_name, device_info) in self._streams_info.items():
      for (stream_name, stream_info) in device_info.items():
        self.clear_data(device_name, stream_name)


  # Get/set metadata
  def get_metadata(self, 
                   device_name: str = None, 
                   only_str_values: bool = False) -> OrderedDict:
    # Get metadata for all devices or for the specified device.
    if device_name is None:
      metadata = self._metadata
    elif device_name in self._metadata:
      metadata = self._metadata[device_name]
    else:
      metadata = OrderedDict()

    # Add the class name.
    class_name = type(self).__name__
    if device_name is None:
      for device_name_toUpdate in metadata:
        metadata[device_name_toUpdate][self.metadata_class_name_key] = class_name
    else:
      metadata[self.metadata_class_name_key] = class_name

    # Convert values to strings if desired.
    if only_str_values:
      metadata = convert_dict_values_to_str(metadata)
    return metadata


  def get_num_devices(self) -> int:
    return len(self._streams_info)


  # Get the names of streaming devices
  def get_device_names(self) -> list[str]:
    return list(self._streams_info.keys())


  # Get the names of streams.
  # If device_name is None, will assume streams are the same for every device.
  def get_stream_names(self, device_name: str = None) -> list[str]:
    if device_name is None:
      device_name = self.get_device_names()[0]
    return list(self._streams_info[device_name].keys())


  # Get information about a stream.
  # Returned dict will have keys:
  #   data_type, 
  #   is_video,
  #   is_audio,
  #   sample_size, 
  #   sampling_rate_hz,
  #   timesteps_before_solidified, 
  #   extra_data_info,
  #   data_notes,
  #   is_measure_rate_hz, (if True)
  #     actual_rate_hz,
  #     dt_circular_buffer,
  #     dt_circular_index,
  #     dt_running_sum,
  #     old_toa
  def get_stream_info(self, device_name: str, stream_name: str) -> OrderedDict[str, Any]:
    return self._streams_info[device_name][stream_name]


  # Get all information about all streams.
  def get_stream_info_all(self) -> OrderedDict[str, OrderedDict[str, OrderedDict[str, Any]]]:
    return copy.deepcopy(self._streams_info)


  # Retrieve actual frame rate of a stream if it was set to measure.
  # Records and refreshes statistics on each data structure append
  #   call, making actual frame rate estimate if data sampled remotely and
  #   sent over LAN to collection device.
  def _get_fps(self, device_name: str, stream_name: str) -> float | None:
    if self._streams_info[device_name][stream_name]['is_measure_rate_hz']:
      return self._streams_info[device_name][stream_name]['actual_rate_hz']
    else:
      return None
