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
import queue
from typing import Any, Callable, Iterable
from collections import OrderedDict, deque


class BufferInterface(ABC):
  @abstractmethod
  def plop(self, key: str, data: dict) -> None:
    pass

  @abstractmethod
  def yeet(self) -> Any:
    pass


class NonOverflowingCounterConverter:
  def __init__(self,
               keys: Iterable,
               num_bits_counter: int):
    self._counter_limit = 2**num_bits_counter
    self._counter_to_nonoverflowing_counter_fn: Callable = self._foo
    self._start_counters = OrderedDict([(k, None) for k in keys])
    self._previous_counters = OrderedDict([(k, None) for k in keys])
    self._counters = OrderedDict([(k, None) for k in keys])
    self._start_counter = None


  def _foo(self, key, counter) -> int | None:
    if self._start_counter is None:
      self._start_counter = counter
      self._start_counters[key] = counter
      self._previous_counters[key] = counter
      self._counters[key] = 0
    elif self._previous_counters[key] is None:
      self._start_counters[key] = counter
      self._previous_counters[key] = counter
      dc = (counter - self._start_counter) % self._counter_limit
      self._counters[key] = dc
    else:
      self._bar(key=key, counter=counter)
    if all([counter is not None for counter in self._start_counters.values()]):
      self._counter_to_nonoverflowing_counter_fn = self._bar
    return self._counters[key]


  def _bar(self, key, counter) -> int | None:
    dc = (counter - self._previous_counters[key]) % self._counter_limit
    self._previous_counters[key] = counter
    self._counters[key] += dc
    return self._counters[key]


class TimestampToCounterConverter:
  def __init__(self,
               keys: Iterable,
               sampling_period: int, # NOTE: sampling period must be in the same units as timestamp limit and timestamps
               num_bits_timestamp: int): # NOTE:
    self._sampling_period = sampling_period
    self._timestamp_limit: int = 2**num_bits_timestamp
    self._counter_from_timestamp_fn: Callable = self._foo
    self._first_timestamps = OrderedDict([(k, None) for k in keys])
    self._previous_timestamps = OrderedDict([(k, None) for k in keys])
    self._counters = OrderedDict([(k, None) for k in keys])


  # Sets the start time according to the first received packet and switches
  #   to the monotone calculation routine after.
  def _foo(self, key, timestamp) -> int | None:
    # The sample is the very first in the buffer -> use as reference timestamp.
    #   Will return 0 start counter at the end of the function.
    if not any([v is not None for v in self._previous_timestamps.values()]):
      self._start_time = timestamp
      self._first_timestamps[key] = timestamp
      self._previous_timestamps[key] = timestamp
      self._counters[key] = 0
    # If it's not the very first packet, but first reading for this device.
    #   Record if the capture was during or after the start reference.
    #   NOTE: style is more verbose to preserve clarity.
    elif self._previous_timestamps[key] is None:
      # Measurement taken during or after the reference measurement and no chance for overflow.
      #   Will return 0 start counter at the end of the function.
      if timestamp >= self._start_time:
        self._first_timestamps[key] = timestamp
        self._previous_timestamps[key] = timestamp
        self._counters[key] = round(((timestamp - self._start_time) % self._timestamp_limit)/ self._sampling_period)
      # Measurement taken after the overflow of the on-sensor clock and effectively after the reference measurement.
      #   Will return 0 start counter at the end of the function.
      elif ((timestamp - self._start_time) % self._timestamp_limit) < (self._start_time - timestamp):
        self._first_timestamps[key] = timestamp
        self._previous_timestamps[key] = timestamp
        self._counters[key] = round(((timestamp - self._start_time) % self._timestamp_limit)/ self._sampling_period)
      # Otherwise it's a stale measurement to be discarded to ensure alignment. 
      else:
        return None
    # Not the first measurement of this device, use the monotone method to compute the counter.
    else:
      self._bar(key=key, timestamp=timestamp)
    # Switch the function call to the monotone routine once all crossed the start reference time.
    if all([v is not None for v in self._previous_timestamps.values()]):
      self._counter_from_timestamp_fn = self._bar
    return self._counters[key]


  def _bar(self, key, timestamp) -> int:
    # Measure the dt between 2 measurements w.r.t. sensor device time and the max value before overlow.
    #   dt > 0 always thanks to modulo, even if sensor on-board clock overflows.
    delta_ticks = (timestamp - self._previous_timestamps[key]) % self._timestamp_limit
    self._previous_timestamps[key] = timestamp
    # Convert to the number of sample periods in that time, allowing for slight instantaneous drift.
    # NOTE: counter measurement with sample rate, previous and current time is more accurate than averaging over whole timelife.
    delta_counter = round(delta_ticks / self._sampling_period)
    self._counters[key] += delta_counter
    return self._counters[key]


# Uses dynamic lists for the buffer, approprate for the sample rate of IMUs.
#   Switch to a defined-length ring buffer to avoid unnecessary memory allocation for higher performance.
class AlignedFifoBuffer(BufferInterface):
  def __init__(self,
               keys: Iterable,
               timesteps_before_stale: int): # NOTE: allows yeeting from buffer if some keys have been empty for a while (disconnection or out of range), while others continue producing
    self._buffer = OrderedDict([(k, deque()) for k in keys])
    self._output_queue = queue.Queue()
    self._counter_snapshot = 0 # Updated only on yeet to discard stale sample that arrived too late.
    self._timesteps_before_stale = timesteps_before_stale


  # Adding packets to the datastructure is asynchronous for each key.
  def plop(self, key: str, data: dict, counter: int):
    # Add counter into the data payload to retreive on the reader. (Useful for time->counter converted buffer).
    data["counter"] = counter
    # The snapshot had not been read yet, even if measurement is stale (arrived later than specified), 
    #   there's still time to add it.
    if counter >= self._counter_snapshot:
      # Empty pad if some intermediate timesteps did not recieve a packet for this key.
      while len(self._buffer[key]) < (counter - self._counter_snapshot):
        self._buffer[key].append(None)
      self._buffer[key].append(data)
    else:
      print("%d packet of %s arrived too late."%(counter, key), flush=True)

    # If buffer contents are valid, move snapshot into the output Queue.
    #   Update frame counter to keep track of removed data to discard stale late arrivals.
    is_every_key_has_data = all([len(buf) for buf in self._buffer.values()])
    is_some_key_exceeds_stale_period = any([len(buf) >= self._timesteps_before_stale for buf in self._buffer.values()])
    is_some_key_empty = any([not len(buf) for buf in self._buffer.values()])
    if is_every_key_has_data:
      oldest_packet = {k: buf.popleft() for k, buf in self._buffer.items()}
      self._put_output_queue(oldest_packet)
    elif is_some_key_exceeds_stale_period and is_some_key_empty:
      oldest_packet = {k: (buf.popleft() if len(buf) else None) for k, buf in self._buffer.items()}
      self._put_output_queue(oldest_packet)


  def _put_output_queue(self, packet: dict) -> None:
    self._counter_snapshot += 1
    self._output_queue.put(packet)


  # No more new data will be captured, can evict all present data.
  def flush(self) -> None:
    while (is_any_key_not_empty := any([len(buf) for buf in self._buffer.values()])):
      oldest_packet = {k: (buf.popleft() if len(buf) else None) for k, buf in self._buffer.items()}
      self._put_output_queue(oldest_packet)


  # Getting packets from the datastructure is synchronous for all keys.
  def yeet(self, timeout: float = 10.0) -> Any | None:
    try:
      return self._output_queue.get(timeout=timeout)
    except queue.Empty:
      print("Timed out on no more snapshots in the output Queue.")
      return None


class TimestampAlignedFifoBuffer(AlignedFifoBuffer):
  def __init__(self,
               keys: Iterable,
               timesteps_before_stale: int, # NOTE: allows yeeting from buffer if some keys have been empty for a while, while others continue producing
               sampling_period: int, # NOTE: sampling period must be in the same units as timestamp limit and timestamps
               num_bits_timestamp: int): # NOTE:
    super().__init__(keys=keys,
                     timesteps_before_stale=timesteps_before_stale)
    self._converter = TimestampToCounterConverter(keys=keys,
                                                  sampling_period=sampling_period,
                                                  num_bits_timestamp=num_bits_timestamp)


  # Override parent method.
  def plop(self, key: str, data: dict, timestamp: float) -> None:
    # Calculate counter from timestamp and local datastructure to avoid race condition.
    counter = self._converter._counter_from_timestamp_fn(key, timestamp)
    if counter is not None:
      super().plop(key=key, data=data, counter=counter)


class NonOverflowingCounterAlignedFifoBuffer(AlignedFifoBuffer):
  def __init__(self,
               keys: Iterable,
               timesteps_before_stale: int,
               num_bits_timestamp: int):
    super().__init__(keys=keys, 
                     timesteps_before_stale=timesteps_before_stale)
    self._converter = NonOverflowingCounterConverter(keys=keys, 
                                                     num_bits_counter=num_bits_timestamp)


  # Override parent method.
  def plop(self, key: str, data: dict, counter: int) -> None:
    # Calculate counter from timestamp and local datastructure to avoid race condition.
    counter = self._converter._counter_to_nonoverflowing_counter_fn(key, counter)
    if counter is not None:
      super().plop(key=key, data=data, counter=counter)
