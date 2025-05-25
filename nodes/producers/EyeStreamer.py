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
from streams import EyeStream
from handlers.PupilLabs.PupilFacade import PupilFacade
from utils.zmq_utils import *
import zmq


#######################################################
#######################################################
# A class to interface with the Pupil Labs eye tracker.
#######################################################
#######################################################
class EyeStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'eye'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               pupil_capture_ip: str = DNS_LOCALHOST,
               pupil_capture_port: str = PORT_EYE,
               video_image_format: str = "bgr", # [bgr, jpeg, yuv]
               gaze_estimate_stale_s: float = 0.2, # how long before a gaze estimate is considered stale (changes color in the world-gaze video)
               is_binocular: bool = True,
               is_stream_video_world: bool = False, 
               is_stream_video_eye: bool = False, 
               is_stream_fixation: bool = False,
               is_stream_blinks: bool = False,
               shape_video_world: tuple[int, int, int] = (1080,720,3),
               shape_video_eye0: tuple[int, int, int] = (192,192,3),
               shape_video_eye1: tuple[int, int, int] = (192,192,3),
               fps_video_world: float = 30.0,
               fps_video_eye0: float = 120.0,
               fps_video_eye1: float = 120.0,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               port_pause: str = PORT_PAUSE,
               timesteps_before_solidified: int = 0,
               **_) -> None:

    self._is_binocular = is_binocular
    self._is_stream_video_world = is_stream_video_world
    self._is_stream_video_eye = is_stream_video_eye
    self._is_stream_fixation = is_stream_fixation
    self._is_stream_blinks = is_stream_blinks

    self._pupil_capture_ip = pupil_capture_ip
    self._pupil_capture_port = pupil_capture_port
    self._video_image_format = video_image_format
    self._gaze_estimate_stale_s = gaze_estimate_stale_s
    self._port_pause = port_pause

    stream_info = {
      "is_binocular": is_binocular,
      "is_stream_video_world": is_stream_video_world,
      "is_stream_video_eye": is_stream_video_eye,
      "is_stream_fixation": is_stream_fixation,
      "is_stream_blinks": is_stream_blinks,
      "gaze_estimate_stale_s": gaze_estimate_stale_s,
      "shape_video_world": shape_video_world,
      "shape_video_eye0": shape_video_eye0,
      "shape_video_eye1": shape_video_eye1,
      "fps_video_world": fps_video_world,
      "fps_video_eye0": fps_video_eye0,
      "fps_video_eye1": fps_video_eye1,
      "pixel_format": video_image_format,
      "timesteps_before_solidified": timesteps_before_solidified
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig)


  @classmethod
  def create_stream(cls, stream_info: dict) -> EyeStream:
    return EyeStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    # TODO: launch Pupil Capture process
    self._handler: PupilFacade = PupilFacade(is_binocular=self._is_binocular,
                                             is_stream_video_world=self._is_stream_video_world,
                                             is_stream_video_eye=self._is_stream_video_eye,
                                             is_stream_fixation=self._is_stream_fixation,
                                             is_stream_blinks=self._is_stream_blinks,
                                             pupil_capture_ip=self._pupil_capture_ip,
                                             pupil_capture_port=self._pupil_capture_port,
                                             gaze_estimate_stale_s=self._gaze_estimate_stale_s,
                                             video_image_format=self._video_image_format)
    self._handler.set_stream_data_getter(fn=self._stream.peek_data_new)
    return True


  def _keep_samples(self) -> None:
    self._handler.keep_data()


  def _process_data(self) -> None:
    res = self._handler.process_data()
    if res is not None:
      process_time_s, data = res
      tag: str = "%s.data" % self._log_source_tag()
      self._publish(tag, process_time_s=process_time_s, data=data)
    elif not self._is_continue_capture:
      self._send_end_packet()


  def _stop_new_data(self):
    self._handler.close()


  def _cleanup(self):
    self._pause.close()
    super()._cleanup()


  ##############################
  ###### Custom Overrides ######
  ##############################
  # For remote pause/resume control.
  # Initialize backend parameters specific to Producer.
  def _initialize(self):
    super()._initialize()
    # Socket to publish sensor data and log
    self._pause: zmq.SyncSocket = self._ctx.socket(zmq.REP)
    self._pause.bind("tcp://*:%s" % (self._port_pause))


  # Listen on the dedicated socket a pause command from the GUI.
  def _activate_data_poller(self) -> None:
    super()._activate_data_poller()
    self._poller.register(self._pause, zmq.POLLIN)


  def _on_poll(self, poll_res):
    if self._pause in poll_res[0]:
      self._pause.recv()
      is_enabled = self._handler.toggle_capturing()
      # NOTE: for now not waiting that the glasses received the message,
      #   but assumes that it happens fast enough before replying 'OK' to the GUI.
      self._pause.send_string(MSG_ON if is_enabled else MSG_OFF)
    super()._on_poll(poll_res)
