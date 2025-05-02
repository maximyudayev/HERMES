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

from collections import OrderedDict, deque
from pypylon import pylon
import numpy as np

from utils.print_utils import *
from utils.time_utils import get_time


# Does not align video streams according to timestamps,
#   they are captured synchronously on independent PoE interfaces -> no need.
#   That's why alignment is not necessary unlike IMUs.
#   Grabbed images from several devices pushed into single buffer, arbitrarily overlapping images.
# NOTE: may be interesting to actually align them in a snapshot buffer, similar to IMUs,
#   To make multi-angle computer vision algorithms possible.
class ImageEventHandler(pylon.ImageEventHandler):
  def __init__(self, cam_array: pylon.InstantCameraArray):
    super().__init__()
    self._is_keep_data = False
    self._cam_array = cam_array
    cam: pylon.InstantCamera
    # Register with the pylon loop, specify strategy for frame grabbing.
    for cam in cam_array:
      cam.RegisterImageEventHandler(self, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
    self._start_sequence_id = OrderedDict([(cam.GetDeviceInfo().GetSerialNumber(), None) for cam in cam_array])
    self._buffer: deque[tuple[str, np.ndarray, bool, int, np.uint64, np.int64, float]] = deque()


  def OnImageGrabbed(self, camera: pylon.InstantCamera, res: pylon.GrabResult):
    # Gets called on every image.
    #   Runs in a pylon thread context, always wrap in the `try .. except`
    #   to capture errors inside the grabbing as this can't be properly
    #   reported from the background thread to the foreground python code.
    try:
      if res.GrabSucceeded():
        if not self._is_keep_data:
          res.Release()
        else:
          toa_s: float = get_time()
          img_buffer = res.GetBuffer()
          camera_id: str = camera.GetDeviceInfo().GetSerialNumber()
          timestamp: np.uint64 = res.GetTimeStamp()
          sequence_id: np.int64 = res.GetImageNumber()
          # Presentation time in the units of the timebase of the stream, w.r.t. the start of the video recording.
          if self._start_sequence_id[camera_id] is None:
            self._start_sequence_id[camera_id] = sequence_id
          pts = sequence_id - self._start_sequence_id[camera_id] # NOTE: not safe against overflow, but int64
          # If there are any skipped images in between, it will take encoder a lot of processing.
          #   Mark the frame as keyframe so it encodes the frame as a whole, not differentially.
          is_keyframe: bool = res.GetNumberOfSkippedImages() > 0
          # Release the buffer for Pylon to reuse for the next frame.
          res.Release()
          # Put the newly allocated converted image into our queue/pipe for Streamer to consume.
          self._buffer.append((camera_id,
                              img_buffer,
                              is_keyframe,
                              pts,
                              timestamp,
                              sequence_id,
                              toa_s))
      else:
        raise RuntimeError("Grab Failed")
    except Exception as e:
      pass


  def get_frame(self) -> tuple[str, np.ndarray, bool, int, np.uint64, np.int64, float] | None:
    try:
      return self._buffer.popleft()
    except IndexError:
      return None


  def keep_data(self):
    self._is_keep_data = True
