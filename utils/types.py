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

from collections import OrderedDict, deque, namedtuple
from typing import TypeAlias, Any, Deque, Iterable, Iterator, Mapping, TypedDict, Dict, NamedTuple
from threading import Lock
import cv2
import zmq


NewDataDict: TypeAlias = Dict[str, Dict[str, Any]]
DataFifo: TypeAlias = Deque[Any]
DataFifoDict: TypeAlias = Dict[str, Dict[str, DataFifo]]
StreamInfoDict: TypeAlias = Dict[str, Dict[str, Dict[str, Any]]]
DeviceLockDict: TypeAlias = Dict[str, Lock]
ExtraDataInfoDict: TypeAlias = Dict[str, Dict[str, Any]]
VideoFormatTuple = namedtuple('VideoFormatTuple', ('ffmpeg_input_format', 'ffmpeg_pix_fmt', 'cv2_cvt_color'))
VideoCodecDict = TypedDict('VideoCodecDict', {'codec_name': str, 'pix_format': str, 'input_options': Mapping, 'output_options': Mapping})
ZMQResult: TypeAlias = Iterable[tuple[zmq.SyncSocket, int]]


# Must be a tuple of (<FFmpeg write format>, <OpenCV display format>):
#   one of the supported FFmpeg pixel formats: https://ffmpeg.org/doxygen/trunk/pixfmt_8h.html#a9a8e335cf3be472042bc9f0cf80cd4c5 
#   one of the supported OpenCV pixel conversion formats: https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
VIDEO_FORMAT = {
  'bgr':        VideoFormatTuple('rawvideo',    'bgr24',        cv2.COLOR_BGR2RGB),
  'yuv':        VideoFormatTuple('rawvideo',    'yuv420p',      cv2.COLOR_YUV2RGB),
  'jpeg':       VideoFormatTuple('image2pipe',  'yuv420p',      cv2.COLOR_YUV2RGB),
  'bayer_rg8':  VideoFormatTuple('rawvideo',    'bayer_rggb8',  cv2.COLOR_BAYER_RG2RGB),
}
