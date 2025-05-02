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
from collections import OrderedDict
from io import TextIOWrapper
from subprocess import Popen
import os
import time
from utils.time_utils import get_time
from typing import Any, Iterator
import wave

import ffmpeg
import asyncio
import concurrent.futures
import h5py
import numpy as np
from streams.Stream import Stream
from utils.dict_utils import convert_dict_values_to_str
from utils.types import VideoCodecDict


##############################################################################################
##############################################################################################
# If using the periodic option, latest data will be fetched from the streamers
#   and then written in a separate thread to avoid undue performance impact.
#   Any leftover data once the program is stopped will be flushed to the files.
# If using the periodic option, can optionally clear old data from memory to reduce RAM usage.
#   Data will then be cleared each time new data is logged, 
#     unless all data is expected to be written at the end.
#   Will treat video/audio data separately, so can choose to stream/clear 
#     non-AV data but dump AV data or vice versa.
# Logging currently supports CSV, HDF5, MP4, and WAV files.
#   If using HDF5, a single file will be created for all the Producers and Pipelines.
#   If using CSV, a separate file will be created for each Producer and Pipeline.
#     N-D data will be unwrapped so that each entry is its own column.
#   Videos can be saved as MP4 files.
#   Audio can be saved as WAV files.
# Note that the is_video / is_audio flags of each stream will be used to identify video/audio.
#   Classes with audio streams will also require a method get_audioStreaming_info()
#     that returns a dict with keys num_channels, sample_width, sampling_rate.
#   See MicrophoneStreamer for an example of defining these attributes.
##############################################################################################
##############################################################################################
class LoggerInterface(ABC):
  @abstractmethod
  def _set_state(self, state) -> None:
    pass

  @abstractmethod
  def _initialize(self, streams: OrderedDict[str, Stream]) -> None:
    pass

  @abstractmethod
  def _log_data(self) -> None:
    pass

  @abstractmethod
  def _is_to_stream(self) -> bool:
    pass

  @abstractmethod
  def _is_to_dump(self) -> bool:
    pass

  @abstractmethod
  def _start_stream_logging(self) -> None:
    pass

  @abstractmethod
  def _stop_stream_logging(self) -> None:
    pass

  @abstractmethod
  def _start_dump_logging(self) -> None:
    pass

  @abstractmethod
  def _wait_till_flush(self) -> None:
    pass

  @abstractmethod
  def _release_thread_pool(self) -> None:
    pass


class BrokerState(ABC):
  def __init__(self, context: LoggerInterface):
    self._context = context
    self._is_continue_fsm = True

  @abstractmethod
  def run(self) -> None:
    pass

  def is_continue(self) -> bool:
    return self._is_continue_fsm


class StartState(BrokerState):
  def __init__(self, context, streams: OrderedDict[str, Stream]):
    super().__init__(context)
    self._context._initialize(streams)

  def run(self) -> None:
    self._context._set_state(StreamState(self._context))


class StreamState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    # Prepare stream-logging.
    if self._context._is_to_stream():
      self._context._start_stream_logging()

  def run(self) -> None:
    asyncio.run(self._context._log_data())
    self._context._release_thread_pool()
    self._context._set_state(DumpState(self._context))


class DumpState(BrokerState):
  def __init__(self, context):
    super().__init__(context)
    # Dump write files at the end of the trial for data that hadn't been streamed.
    #   Assumes all intermediate recorded data can fit in memory.
    if self._context._is_to_dump():
      self._context._start_dump_logging()

  def run(self) -> None:
    # Until top-level module's main thread indicated that it finished producing data,
    #   periodically sleep the thread to yield the CPU.
    self._context._wait_till_flush()
    # TODO: When experiment ended, write data once and wrap up.
    #   Use the end log time to not overwrite written streamed data with empty dumped files?
    asyncio.run(self._context._log_data())
    # Release Thread Pool used for AsyncIO for file writing.
    self._context._release_thread_pool()
    self._is_continue_fsm = False


class Logger(LoggerInterface):
  def __init__(self,
               log_tag: str,
               log_dir: str,
               stream_csv: bool = False,
               stream_hdf5: bool = False,
               stream_video: bool = False,
               stream_audio: bool = False,
               dump_csv: bool = False,
               dump_hdf5: bool = False,
               dump_video: bool = False,
               dump_audio: bool = False,
               video_codec: VideoCodecDict = None,
               video_codec_num_cpu: int = 1,
               audio_format: str = "wav",
               stream_period_s: float = 30.0,
               **_):

    # Record the configuration options.
    self._stream_hdf5 = stream_hdf5
    self._stream_csv = stream_csv
    self._stream_video = stream_video
    self._stream_audio = stream_audio
    self._stream_period_s = stream_period_s
    self._dump_hdf5 = dump_hdf5
    self._dump_csv = dump_csv
    self._dump_video = dump_video
    self._dump_audio = dump_audio
    self._video_codec = video_codec
    self._video_codec_num_cpu = video_codec_num_cpu
    self._audio_format = audio_format
    self._log_tag = log_tag
    self._log_dir = log_dir
    
    # Initialize the logging writers.
    self._thread_pool: concurrent.futures.ThreadPoolExecutor
    self._hdf5_file: h5py.File = None
    self._video_writers: list[tuple[Popen, str, str, str]] = []
    self._audio_writers: list[tuple[wave.Wave_write, str, str, str]] = []
    self._csv_writers: list[tuple[TextIOWrapper, str, str, str]] = []
    self._csv_writer_metadata: TextIOWrapper = None
  
    # Create the log directory if needed.
    if self._stream_csv or self._stream_hdf5 or self._stream_video or self._stream_audio \
        or self._dump_csv or self._dump_hdf5 or self._dump_video or self._dump_audio:
      os.makedirs(self._log_dir, exist_ok=True)

    # Initialize variables that will guide the thread that will do stream/dump logging of data available in the Stream objects.
    #   Main thread will listen to the sockets and put files to the Stream objects.
    self._is_streaming = None   # whether periodic writing is active
    self._is_flush = None       # whether remaining data at the end should now be flushed
    self._is_finished = None    # whether the logging loop is finished and all data was flushed


  def __call__(self, streams: OrderedDict[str, Stream]) -> None:
    self._state = StartState(self, streams)
    # Run continuously, ignoring Ctrl+C interrupt, until owner Node commands an exit.
    while self._state.is_continue():
      self._state.run()
    print("%s Logger safely exited."%self._log_tag, flush=True)


  # Stop logging.
  # Will stop stream-logging, if it is active.
  # Will trigger data dump, if configured.
  # Node pushing data to the Stream should stop adding new data before cleaning up Logger. 
  def cleanup(self) -> None:
    # Stop stream-logging and wait for it to finish.
    self._stop_stream_logging()
    self._log_stop_time_s = get_time()


  ############################
  ###### FSM OPERATIONS ######
  ############################
  # Initializes files and indices for write pointer tracking.
  def _initialize(self, streams: OrderedDict[str, Stream]) -> None:
    # Create Stream objects for all desired sensors we are to subscribe to from classes_to_log
    self._hdf5_log_length_increment = 10000
    self._streams = streams
    self._timesteps_before_solidified: OrderedDict[str, OrderedDict[str, OrderedDict[str, int]]] = OrderedDict()
    self._next_data_indices_hdf5: OrderedDict[str, OrderedDict[str, OrderedDict[str, int]]] = OrderedDict()
    for tag in streams.keys():
      # Initialize a record of what indices have been logged,
      #  and how many timesteps to stay behind of the most recent step (if needed).
      self._timesteps_before_solidified.setdefault(tag, OrderedDict())
      # Each time an HDF5 dataset reaches its limit,
      #  its size will be increased by the following amount.
      self._next_data_indices_hdf5.setdefault(tag, OrderedDict())


  def _set_state(self, state: BrokerState) -> None:
    self._state = state


  def _is_to_stream(self) -> bool:
    return self._stream_csv or self._stream_hdf5 or self._stream_video or self._stream_audio


  def _is_to_dump(self) -> bool:
    return self._dump_csv or self._dump_hdf5 or self._dump_video or self._dump_audio


  def _start_stream_logging(self) -> None:
    # Set up CSV/HDF5 file writers for stream-logging if desired.
    num_workers: int = 0
    if self._stream_csv:
      num_workers += self._init_files_csv()
    if self._stream_hdf5:
      num_workers += self._init_files_hdf5()
    if self._stream_video:
      num_workers += self._init_files_video()
    if self._stream_audio:
      num_workers += self._init_files_audio()
    self._init_log_indices()
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    self._is_streaming = True
    self._is_flush = False
    self._is_finished = False


  # Helper to stop the stream-logging thread to periodically write data.
  # Will write any outstanding data,
  #  and will log any metadata associated with the streamers.
  # Will wait for the thread to finish before returning.
  def _stop_stream_logging(self) -> None:
    self._is_streaming = False
    self._is_flush = True


  def _start_dump_logging(self) -> None:
    num_workers: int = 0
    if self._dump_csv:
      num_workers += self._init_files_csv()
    if self._dump_hdf5:
      num_workers += self._init_files_hdf5()
    if self._dump_video:
      num_workers += self._init_files_video()
    if self._dump_audio:
      num_workers += self._init_files_audio()
    # Log all data.
    # Will basically enable periodic stream-logging,
    #  but will set self._is_flush and self._is_streaming such that
    #  it seems like the experiment ended and just a final flush is required.
    #  This will cause the stream-logging in self._log_data()
    #  to fetch and write any outstanding data, which is all data since
    #  none is written yet.  It will then exit after the single write.
    # Pretend like the dumping options are actually streaming options.
    self._stream_csv = self._dump_csv
    self._stream_hdf5 = self._dump_hdf5
    self._stream_video = self._dump_video
    self._stream_audio = self._dump_audio
    # Clear the is_finished flag in case dump is run after stream so the log loop can run once to flush all logged data that wasn't stream-logged.
    self._is_finished = False
    # Initialize indexes and log all of the data.
    self._init_log_indices()
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)


  def _wait_till_flush(self) -> None:
    while not self._is_flush: 
      time.sleep(self._stream_period_s)


  def _release_thread_pool(self) -> None:
    self._thread_pool.shutdown()


  #############################
  ###### FILE OPERATIONS ######
  #############################
  # Initialize the data indices to fetch for logging.
  # Will record the next data indices that should be fetched for each stream,
  #  and the number of timesteps that each streamer needs before data is solidified.
  def _init_log_indices(self) -> None:
    for (streamer_name, stream) in self._streams.items():
      for (device_name, device_info) in stream.get_stream_info_all().items():
        self._timesteps_before_solidified[streamer_name][device_name] = OrderedDict()
        for (stream_name, stream_info) in device_info.items():
          self._timesteps_before_solidified[streamer_name][device_name][stream_name] = stream_info['timesteps_before_solidified']


  # Create and initialize CSV files.
  # Will have a separate file for each stream of each device.
  # Currently assumes that device names are unique across all streamers.
  def _init_files_csv(self) -> int:
    # Open a writer for each CSV data file.
    num_writers: int = 0
    self._csv_writers = OrderedDict([(k, OrderedDict()) for k in self._streams.keys()])
    for (streamer_name, streamer) in self._streams.items():
      for (device_name, device_info) in streamer.get_stream_info_all().items():
        self._csv_writers[streamer_name][device_name] = OrderedDict()
        for (stream_name, stream_info) in device_info.items():
          # Skip saving video or audio in a CSV.
          if stream_info['is_video'] or stream_info['is_audio']:
            continue
          filename_csv = '%s_%s_%s.csv' % (self._log_tag, device_name, stream_name)
          filepath_csv = os.path.join(self._log_dir, filename_csv)
          csv_writer = open(filepath_csv, 'w')
          self._csv_writers.append(csv_writer, streamer_name, device_name, stream_name)
          num_writers += 1

    # Open a writer for a CSV metadata file.
    filename_csv = '%s__metadata.csv' % (self._log_tag)
    filepath_csv = os.path.join(self._log_dir, filename_csv)
    self._csv_writer_metadata = open(filepath_csv, 'w')

    # Write CSV headers.
    for (stream_writer, streamer_name, device_name, stream_name) in self._csv_writers:
      # First check if custom header titles have been specified.
      stream_info = streamer.get_stream_info_all()[device_name][stream_name]
      sample_size = stream_info['sample_size']
      if isinstance(stream_info['data_notes'], dict) and Stream.metadata_data_headings_key in stream_info['data_notes']:
        data_headers = stream_info['data_notes'][Stream.metadata_data_headings_key]
      else:
        # Write a number of data headers based on how many values are in each data sample.
        # Each sample may be a matrix that will be unwrapped into columns,
        #  so label headers as i-j where i is the original matrix row
        #  and j is the original matrix column (and if more than 2D keep adding more).
        data_headers = []
        subs = np.unravel_index(range(0,np.prod(sample_size)), sample_size)
        subs = np.stack(subs).T
        for header_index in range(subs.shape[0]):
          header = 'Data Entry '
          for sub_index in range(subs.shape[1]):
            header += '%d-' % subs[header_index, sub_index]
          header = header.strip('-')
          data_headers.append(header)
      stream_writer.write(',')
      stream_writer.write(','.join(data_headers))
    return num_writers


  # Create and initialize an HDF5 file.
  # Will have a single file for all streams from all devices.
  # Currently assumes that device names are unique across all streamers.
  def _init_files_hdf5(self) -> int:
    # Open an HDF5 file writerю
    filename_hdf5 = '%s.hdf5' % self._log_tag
    filepath_hdf5 = os.path.join(self._log_dir, filename_hdf5)
    num_to_append = 0
    while os.path.exists(filepath_hdf5):
      num_to_append += 1
      filename_hdf5 = '%s_%02d.hdf5' % (self._log_tag, num_to_append)
      filepath_hdf5 = os.path.join(self._log_dir, filename_hdf5)
    self._hdf5_file = h5py.File(filepath_hdf5, 'w')
    # Create a dataset for each data key of each stream of each device.
    for (streamer_name, stream) in self._streams.items():
      for (device_name, device_info) in stream.get_stream_info_all().items():
        device_group = self._hdf5_file.create_group(device_name)
        self._next_data_indices_hdf5[streamer_name][device_name] = OrderedDict()
        for (stream_name, stream_info) in device_info.items():
          # Skip saving video and audio in the HDF5.
          if stream_info['is_video'] or stream_info['is_audio']:
            continue
          self._next_data_indices_hdf5[streamer_name][device_name][stream_name] = 0
          # The main data has specifications defined by stream_info.
          sample_size = stream_info['sample_size']
          data_type = stream_info['data_type']
          # Create the dataset.
          device_group.create_dataset(name=stream_name,
                                      shape=(self._hdf5_log_length_increment, *sample_size),
                                      maxshape=(None, *sample_size),
                                      dtype=data_type,
                                      chunks=True)
    return 1


  # Create and initialize video writers.
  def _init_files_video(self) -> int:
    # Create a video writer for each video stream of each device.
    num_writers: int = 0
    for (streamer_name, streamer) in self._streams.items():
      for (device_name, device_info) in streamer.get_stream_info_all().items():
        for (stream_name, stream_info) in device_info.items():
          # Skip non-video streams.
          if not stream_info['is_video']:
            continue
          # Create a unique file.
          filename_base = '%s_%s' % (self._log_tag, device_name)
          filename_video = '%s.mp4' % (filename_base)
          filepath_video = os.path.join(self._log_dir, filename_video)
          num_to_append = 0
          while os.path.exists(filepath_video):
            num_to_append += 1
            filename_video = '%s_%02d.mp4' % (filename_base, num_to_append)
            filepath_video = os.path.join(self._log_dir, filename_video)
          # Create a video writer.
          frame_height = stream_info['sample_size'][0]
          frame_width = stream_info['sample_size'][1]
          fps = stream_info['sampling_rate_hz']
          input_stream_pix_fmt: str = stream_info['color_format']['ffmpeg']
          input_stream_format: str = stream_info['ffmpeg_input_format']

          # Make a subprocess pipe to FFMPEG that streams in our frames and encode them into a video.
          # TODO: adjust the stream specification to use the `is_keyframe` and `pts` when providing frame to ffmpeg.
          # ffmpeg.setpts()
          video_writer: Popen = (
            ffmpeg
            .input('pipe:',
                   format=input_stream_format,
                   pix_fmt=input_stream_pix_fmt, # color format of piped input frames.
                   s='{}x{}'.format(frame_width, frame_height), # size of frames from the sensor.
                   framerate=fps,
                   cpucount=self._video_codec_num_cpu
                   )
            # .filter('scale', 1280, 720)
            # .filter('deflicker', mode='pm', size=10) # remove line noise from room lights.
            .output(filepath_video,
                    vcodec=self._video_codec['codec_name'],
                    pix_fmt=self._video_codec['pix_format'],
                    cpucount=self._video_codec_num_cpu, # prevent ffmpeg from suffocating the processor.                    
                    **self._video_codec['options']
                    )
            .run_async(pipe_stdin=True)
          )
          # Store the writer.
          self._video_writers.append((video_writer, streamer_name, device_name, stream_name))
          num_writers += 1
    return num_writers


  # Create and initialize audio writers.
  # TODO: implement audio streaming info on the Stream object.
  # TODO: switch to ffmpeg for audio file writing.
  def _init_files_audio(self) -> int:
    # Create an audio writer for each audio stream of each device.
    num_writers: int = 0
    self._audio_writers = OrderedDict([(k, OrderedDict()) for k in self._streams.keys()])
    for (streamer_name, stream) in self._streams.items():
      for (device_name, streams_info) in stream.get_stream_info_all().items():
        for (stream_name, stream_info) in streams_info.items():
          # Skip non-audio streams.
          if not stream_info['is_audio']:
            continue
          # Create an audio writer.
          filename_base = '%s_%s' % (self._log_tag, device_name)
          filename_audio = '%s.wav' % (filename_base)
          filepath_audio = os.path.join(self._log_dir, filename_audio)
          num_to_append = 0
          while os.path.exists(filepath_audio):
            num_to_append += 1
            filename_audio = '%s_%02d.wav' % (filename_base, num_to_append)
            filepath_audio = os.path.join(self._log_dir, filename_audio)
          audio_writer = wave.open(filepath_audio, 'wb')
          # TODO: implement this Stream method in the AudioStream class.
          # stream: AudioStream
          audio_streaming_info = stream.get_audio_streaming_info()
          audio_writer.setnchannels(audio_streaming_info['num_channels'])
          audio_writer.setsampwidth(audio_streaming_info['sample_width'])
          audio_writer.setframerate(audio_streaming_info['sampling_rate'])
          # Store the writer.
          self._audio_writers.append((audio_writer, streamer_name, device_name, stream_name))
          num_writers += 1
    return num_writers


  def _log_metadata_csv(self) -> None:
    for (streamer_name, stream) in self._streams.items():
      for (device_name, device_info) in stream.get_stream_info_all().items():
        # Get metadata for this device.
        # To make it HDF5 compatible,
        #  flatten the dictionary and then
        #  prune objects that can't be converted to a string easily.
        device_metadata = stream.get_metadata(device_name=device_name, only_str_values=True)
        device_metadata = convert_dict_values_to_str(device_metadata, preserve_nested_dicts=False)
        if self._csv_writer_metadata is not None:
          self._csv_writer_metadata.write('\n%s,%s\n' % ('='*25,'='*25))
          self._csv_writer_metadata.write('Device Name,%s' % (device_name))
          self._csv_writer_metadata.write('\n%s,%s' % ('='*25,'='*25))
          for (meta_key, meta_value) in device_metadata.items():
            self._csv_writer_metadata.write('\n')
            self._csv_writer_metadata.write('%s,"%s"' % (str(meta_key), str(meta_value)))
          self._csv_writer_metadata.write('\n')
        # Get data notes for each stream.
        for (stream_name, stream_info) in device_info.items():
          data_notes = stream_info['data_notes']
          if isinstance(data_notes, dict):
            stream_metadata = data_notes
          else:
            stream_metadata = {'data_notes': data_notes}
          stream_metadata = convert_dict_values_to_str(stream_metadata, preserve_nested_dicts=False)
          # Write the stream-level metadata.
          if self._csv_writer_metadata is not None:
            self._csv_writer_metadata.write('\n')
            self._csv_writer_metadata.write('Stream Name,%s' % (stream_name))
            for (meta_key, meta_value) in stream_metadata.items():
              self._csv_writer_metadata.write('\n')
              self._csv_writer_metadata.write('%s,"%s"' % (str(meta_key), str(meta_value)))
            self._csv_writer_metadata.write('\n')


  def _log_metadata_hdf5(self) -> None:
    for (streamer_name, stream) in self._streams.items():
      for (device_name, device_info) in stream.get_stream_info_all().items():
        # Get metadata for this device.
        # To make it HDF5 compatible,
        #  flatten the dictionary and then
        #  prune objects that can't be converted to a string easily.
        device_metadata = stream.get_metadata(device_name=device_name, only_str_values=True)
        device_metadata = convert_dict_values_to_str(device_metadata, preserve_nested_dicts=False)
        # Write the device-level metadata.
        if self._hdf5_file is not None:
          device_group = self._hdf5_file['/'.join([device_name])]
          device_group.attrs.update(device_metadata)
        # Get data notes for each stream.
        for (stream_name, stream_info) in device_info.items():
          data_notes = stream_info['data_notes']
          if isinstance(data_notes, dict):
            stream_metadata = data_notes
          else:
            stream_metadata = {'data_notes': data_notes}
          stream_metadata = convert_dict_values_to_str(stream_metadata, preserve_nested_dicts=False)
          if self._hdf5_file is not None:
            try:
              stream_group = self._hdf5_file['/'.join([device_name, stream_name])]
              stream_group.attrs.update(stream_metadata)
            except KeyError: # a writer was not created for this stream
              pass


  # TODO: provide metadata on video.
  def _log_metadata_video(self) -> None:
    pass


  # TODO: provide metadata on audio.
  def _log_metadata_audio(self) -> None:
    pass


  # Write metadata from each streamer to CSV and/or HDF5 files.
  # Will include device-level metadata and any lower-level data notes.
  def _log_metadata(self):
    self._log_metadata_csv()
    self._log_metadata_hdf5()
    self._log_metadata_video()
    self._log_metadata_audio()


  # Flush/close the HDF5 file writer.
  def _close_files_hdf5(self) -> None:
    # Also resize datasets to remove extra empty rows.
    if self._hdf5_file is not None:
      for (streamer_name, stream) in self._streams.items():
        for (device_name, device_info) in stream.get_stream_info_all().items():
          for (stream_name, stream_info) in device_info.items():
            try:
              dataset: h5py.Dataset = self._hdf5_file['/'.join([device_name, stream_name])]
            except KeyError: # a dataset was not created for this stream
              continue
            starting_index = self._next_data_indices_hdf5[streamer_name][device_name][stream_name]
            ending_index = starting_index - 1
            dataset.resize((ending_index+1, *dataset.shape[1:]))
      self._hdf5_file.close()
      self._hdf5_file = None


  # Flush/close all of the video writers.
  def _close_files_video(self) -> None:
    for (video_writer, *_) in self._video_writers:
      video_writer.stdin.close()
      video_writer.wait()
    self._video_writers = []


  # Flush/close all of the CSV writers.
  def _close_files_csv(self) -> None:
    for (stream_writer, *_) in self._csv_writers:
      stream_writer.close()
    self._csv_writers = []
    if self._csv_writer_metadata is not None:
      self._csv_writer_metadata.close()
      self._csv_writer_metadata = None


  # Flush/close all of the audio writers.
  def _close_files_audio(self) -> None:
    for (audio_writer, *_) in self._audio_writers:
      audio_writer.close()
    self._audio_writers = []


  # Flush/close CSV, HDF5, and video file writers.
  def _close_files(self) -> None:
    self._close_files_csv()
    self._close_files_hdf5()
    self._close_files_video()
    self._close_files_audio()


  # Write provided data to the HDF5 file.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  def _sync_write_hdf5(self, 
                       streamer_name: str, 
                       device_name: str, 
                       stream_name: str) -> None:
    try:
      dataset: h5py.Dataset = self._hdf5_file['/'.join([device_name, stream_name])]
    except KeyError: # a dataset was not created for this stream
      return
    new_data: Iterator[Any] = self._streams[streamer_name].pop_data(device_name=device_name, stream_name=stream_name, is_flush=self._is_flush)
    # Write all available data to HDF5 file.
    for data_to_write in new_data:
      # Extend the dataset as needed while iterating over the 'new_data'.
      starting_index = self._next_data_indices_hdf5[streamer_name][device_name][stream_name]
      # Expand the dataset if needed.
      if not (starting_index < len(dataset)):
        dataset.resize((len(dataset) + self._hdf5_log_length_increment, *dataset.shape[1:]))
      # If data is a string.
      dataset_dtype: np.dtype = dataset.dtype
      if dataset_dtype.char == 'S':
        data_to_write: str
        data_to_write = [data_to_write.encode("ascii", "ignore")]
      # Write the new entries.
      dataset[starting_index,:] = np.array(data_to_write).reshape((-1, *dataset.shape[1:]))
      # Update the next starting index to use.
      starting_index += 1
      self._next_data_indices_hdf5[streamer_name][device_name][stream_name] = starting_index
    # Flush the file with the new data.
    self._hdf5_file.flush()


  # Write provided data to the video files.
  # Note that this can be called during streaming (periodic writing)
  #   or during post-experiment dumping.
  def _sync_write_video(self,
                        video_writer: Popen,
                        streamer_name: str,
                        device_name: str,
                        stream_name: str):
    # Write all available video frames to file.
    new_data: Iterator[tuple[bytes, bool, int]] = self._streams[streamer_name].pop_data(device_name=device_name, 
                                                                                        stream_name=stream_name, 
                                                                                        is_flush=self._is_flush)
    for buffer, is_keyframe, pts in new_data:
      # TODO: adjust the stream specification to use the `is_keyframe` and `pts` when providing frame to ffmpeg.
      video_writer.stdin.write(buffer)


  # Write provided data to the CSV file.
  # Note that this can be called during streaming (periodic writing)
  #  or during post-experiment dumping.
  def _sync_write_csv(self,
                      stream_writer: TextIOWrapper,
                      streamer_name: str, 
                      device_name: str, 
                      stream_name: str) -> None:
    new_data: Iterator[Any] = self._streams[streamer_name].pop_data(device_name=device_name, stream_name=stream_name, is_flush=self._is_flush)
    # Write all available data to CSV file.
    for data_to_write in new_data:
      # Create a list of column entries to write.
      # Note that they should match the heading order in _init_writing_csv().
      if isinstance(data_to_write, np.ndarray):
        to_write = list(np.atleast_1d(data_to_write.reshape(1, -1).squeeze()))
      elif isinstance(data_to_write, (list, tuple)):
        to_write = data_to_write
      else:
        to_write = list(data_to_write)
      # Write the new row.
      stream_writer.write('\n')
      stream_writer.write(','.join([str(x) for x in to_write]))
    stream_writer.flush()


  # Write provided data to the audio files.
  # Note that this can be called during streaming (periodic writing)
  #   or during post-experiment dumping.
  def _sync_write_audio(self,
                        audio_writer: wave.Wave_write,
                        streamer_name: str, 
                        device_name: str, 
                        stream_name: str):
    new_data: Iterator[Any] = self._streams[streamer_name].pop_data(device_name=device_name, stream_name=stream_name, is_flush=self._is_flush)
    # Write all available audio frames to file.
    for frame in new_data:
      # Assume the data is a list of lists (each entry is a list of chunked audio data).
      audio_writer.writeframes(bytearray(frame))


  # Wraps writing of multiple asynchronous Stream Deque data structures 
  #   into another synchronous routine to write all text Stream data to a single HDF5 file.
  def _write_hdf5(self) -> None:
    # Write new data for each stream of each device of each streamer.
    for (streamer_name, stream) in self._streams.items():
      for (device_name, device_info) in stream.get_stream_info_all().items():
        for (stream_name, stream_info) in device_info.items():
          self._sync_write_hdf5(streamer_name=streamer_name, 
                                device_name=device_name, 
                                stream_name=stream_name)


  # Wraps synchronous writing of multiple asynchronous Stream Deque data structures 
  #   into an asynchronous coroutine used to concurrently write multiple video files.
  async def _write_video(self,
                         video_writer: Popen,
                         streamer_name: str,
                         device_name: str,
                         stream_name: str):
    await asyncio.get_event_loop().run_in_executor(
      self._thread_pool,
      lambda: self._sync_write_video(video_writer=video_writer,
                                     streamer_name=streamer_name, 
                                     device_name=device_name, 
                                     stream_name=stream_name))


  # Wraps synchronous writing of multiple asynchronous Stream Deque data structures 
  #   into an asynchronous coroutine used to concurrently write multiple CSV files.
  async def _write_csv(self,
                       stream_writer: TextIOWrapper,
                       streamer_name: str,
                       device_name: str,
                       stream_name: str):
    await asyncio.get_event_loop().run_in_executor(
      self._thread_pool,
      lambda: self._sync_write_csv(stream_writer=stream_writer,
                                   streamer_name=streamer_name, 
                                   device_name=device_name, 
                                   stream_name=stream_name))


  # Wraps synchronous writing of multiple asynchronous Stream Deque data structures 
  #   into an asynchronous coroutine used to concurrently write multiple audio files.
  async def _write_audio(self,
                         audio_writer: wave.Wave_write,
                         streamer_name: str,
                         device_name: str,
                         stream_name: str):
    await asyncio.get_event_loop().run_in_executor(
      self._thread_pool,
      lambda: self._sync_write_audio(audio_writer=audio_writer,
                                     streamer_name=streamer_name, 
                                     device_name=device_name, 
                                     stream_name=stream_name))


  # Wraps synchronous writing to a single HDF5 file, 
  #   from multiple asynchronous Stream Deque data structures,
  #   into an asynchronous coroutine that can be run concurrently to other file IO.
  async def _write_files_hdf5(self):
    await asyncio.get_event_loop().run_in_executor(
      self._thread_pool,
      lambda: self._write_hdf5())


  # Awaits completion from a wrapper that wraps concurrent writing to multiple video files,
  #   of multiple asynchronous Stream Deque data structures containing video data.
  async def _write_files_video(self):
    tasks = []
    # Write new data for each stream of each device of each streamer.
    for (video_writer, streamer_name, device_name, stream_name) in self._video_writers:
      tasks.append(
        self._write_video(video_writer=video_writer,
                          streamer_name=streamer_name, 
                          device_name=device_name, 
                          stream_name=stream_name))
    await asyncio.gather(*tasks) 


  # Awaits completion from a wrapper that wraps concurrent writing to multiple CSV files,
  #   of multiple asynchronous Stream Deque data structures containing text data.
  async def _write_files_csv(self):
    tasks = []
    # Write new data for each stream of each device of each streamer.
    for (stream_writer, streamer_name, device_name, stream_name) in self._csv_writers:
      tasks.append(
        self._write_csv(stream_writer=stream_writer,
                        streamer_name=streamer_name, 
                        device_name=device_name, 
                        stream_name=stream_name))
    await asyncio.gather(*tasks)
    

  # Awaits completion from a wrapper that wraps concurrent writing to multiple audio files,
  #   of multiple asynchronous Stream Deque data structures containing audio data.
  async def _write_files_audio(self):
    tasks = []
    # Write new data for each stream of each device of each streamer.
    for (audio_writer, streamer_name, device_name, stream_name) in self._audio_writers:
      tasks.append(
        self._write_audio(audio_writer=audio_writer,
                          streamer_name=streamer_name, 
                          device_name=device_name, 
                          stream_name=stream_name))
    await asyncio.gather(*tasks) 


  ##########################
  ###### DATA LOGGING ######
  ##########################
  # Poll data from each streamer and log it, either periodically or all at once.
  # The poll period is set by self._stream_period_s.
  # Will loop until self._is_streaming is False, and then
  #  will do one final fetch/log if self._is_flush is True.
  # Usage to periodically poll data from each streamer and log it:
  #   Set self._is_streaming to True and self._is_flush to False
  #     then run this method in a thread.
  #   To finish, set self._is_streaming to False and self._is_flush to True.
  # Usage to log all available data once:
  #   Set self._is_streaming to False and self._is_flush to True
  #     then run this method (in a thread or in the main thread).
  async def _log_data(self) -> None:
    # Used to run periodic data writing.
    last_log_time_s = None
    # Set at the beginning of the iteration if _is_flush is externally modified to indicate cleanup and exit,
    #   to catch case where external command to flush happened while some of streamers already saved part of available data.
    is_flush_all_in_current_iteration = False
    while (self._is_streaming or self._is_flush) and not self._is_finished:
      # Wait until it is time to write new data, which is either:
      #  1. This is the first iteration.
      #  2. It has been at least self._stream_period_s since the last write.
      #  3. Periodic logging has been deactivated.
      while (last_log_time_s is not None
            and (time_to_next_period := (last_log_time_s + self._stream_period_s - get_time())) > 0
            and self._is_streaming):
        # Will wake up periodically to check if the experiment had been ended.
        #   Will proceed only if time for next logging or if experiment ended.
        await asyncio.sleep(min(1, time_to_next_period))
      # If running Logger in dump mode, wait until _is_flush is set externally.
      if not self._is_streaming and not self._is_flush:
        continue
      # Update the last log time now, before the write actually starts.
      # This will keep the log period more consistent; otherwise, the amount
      #   of time it takes to perform the write would be added to the log period.
      #   This would compound over time, leading to longer delays and more data to write each time.
      #   This becomes more severe as the write duration increases (e.g. videos).
      last_log_time_s = get_time()
      # If the log should be flushed, record that it is happening during this iteration for ALL streamers.
      if self._is_flush:
        is_flush_all_in_current_iteration = True
      # Delegate file writing to each AsyncIO method that manages corresponding stream type writing.
      tasks = []
      if self._stream_hdf5:
        tasks.append(self._write_files_hdf5())
      if self._stream_video:
        tasks.append(self._write_files_video())
      if self._stream_csv:
        tasks.append(self._write_files_csv())
      if self._stream_audio:
        tasks.append(self._write_files_audio())
      # Execute all file writing concurrently.
      await asyncio.gather(*tasks)
      # If stream-logging is disabled, but a final flush had been requested,
      #   record that the flush is complete so streaming can really stop now.
      # Note that it also checks whether the flush was configured to happen for all streamers during this iteration.
      #   Especially if a lot of data was being written (such as with video data),
      #   the self._is_flush flag may have been set sometime during the data writing.
      #   In that case, all streamers would not have known to flush data and some data may be omitted.
      # flushing_log set True when _is_flush was set before any streamer saved its data chunk, to make ure nothing is left behind. 
      if (not self._is_streaming) and self._is_flush and is_flush_all_in_current_iteration:
        self._is_finished = True
    # Log metadata.
    self._log_metadata()
    # Save and close the files.
    self._close_files()
