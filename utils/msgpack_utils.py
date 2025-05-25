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

import msgpack
import numpy as np


def encode_ndarray(obj):
  if isinstance(obj, np.ndarray):
    return {'__numpy__': True, 
            'shape': obj.shape, 
            'dtype': str(obj.dtype), 
            'bytes': obj.tobytes()}
  return obj


def decode_ndarray(obj):
  if '__numpy__' in obj:
    obj = np.frombuffer(obj['bytes'], dtype=obj['dtype']).reshape(obj['shape'])
  return obj


# Serializes the message objects.
#   Preserves named arguments as key-value pairs for a dictionary-like message.
def serialize(**kwargs) -> bytes:
  return msgpack.packb(o=kwargs, default=encode_ndarray) # type: ignore


# Deserializes the message back into a dictionary-like message.
def deserialize(msg) -> dict:
  return msgpack.unpackb(msg, object_hook=decode_ndarray)
