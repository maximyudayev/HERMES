############
#
# Copyright (c) 2025 Vayalet Stefanova and KU Leuven eMedia Lab
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
# Created 2025 for AidWear, AidFOG, and RevalExo projects of KU Leuven.
#
# ############

import numpy as np

class DotsCollector:
    """
    Wraps movella_dot_py.core.collector.DataCollector.
    Adds gyroscope & magnetometer storage + getters,
    but forwards everything else to the original collector.
    """

    def __init__(self, inner):
        self._inner = inner
        # keep the parser used by the SDK
        self.parser = inner.parser

        self._gyroscope = [] 
        self._magnetometer  = []      

    def add_data(self, raw: bytes):
        pd = self.parser.parse(raw)
        gyro = getattr(pd, "angular_velocity", None)
        magn = getattr(pd, "magnetic_field", None)

        if gyro is not None:
            self._gyroscope.append([float(gyro.x), float(gyro.y), float(gyro.z)])
        if magn is not None:
            self._magnetometer.append([float(magn.x), float(magn.y), float(magn.z)])

        return self._inner.add_data(raw)
    
    def get_gyroscopes(self) -> np.ndarray:
        return np.asarray(self._gyroscope, dtype=float) if self._gyroscope else np.empty((0, 3), dtype=float)

    def get_magnetometers(self) -> np.ndarray:
        return np.asarray(self._magnetometer, dtype=float) if self._magnetometer else np.empty((0, 3), dtype=float)

    # pass-through for everything the SDK already exposes 
    def get_timestamps(self):    return self._inner.get_timestamps()
    def get_quaternions(self):   return self._inner.get_quaternions()
    def get_euler_angles(self):  return self._inner.get_euler_angles()
    def get_accelerations(self): return self._inner.get_accelerations()

    def get_collected_data(self):
        out = {
            'mac_address': self.mac_address,
            "timestamps": self.get_timestamps(),
            "quaternions": self.get_quaternions(),
            "euler_angles": self.get_euler_angles(),
            "acceleration": self.get_accelerations(),
            "gyroscope": self.get_gyroscopes(),
            "magnetometer": self.get_magnetometers()
        }
        return {k: v for k, v in out.items() if v is not None}

    def __getattr__(self, name):
        return getattr(self._inner, name)