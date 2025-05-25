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

from typing import Any, Callable, Iterable, Mapping, TypedDict
from utils.user_settings import *
import movelladot_pc_sdk as mdda

import numpy as np


MovellaDataGetter = TypedDict('MovellaDataGetter', {'func': Callable[[mdda.XsDataPacket], Any], 'n_dim': tuple, 'dtype': type, 'type_str': str}) # type: ignore
MovellaPayloadTuple = TypedDict('MovellaPayloadTuple', {'num_bytes': int, 'payload_mode': Any, 'methods': Mapping[str, MovellaDataGetter]})


class MovellaPayloadEnum:
  EXTENDED_QUATERNION = mdda.XsPayloadMode_ExtendedQuaternion # type: ignore
  COMPLETE_QUATERNION = mdda.XsPayloadMode_CompleteQuaternion # type: ignore
  EXTENDED_EULER = mdda.XsPayloadMode_ExtendedEuler # type: ignore
  COMPLETE_EULER = mdda.XsPayloadMode_CompleteEuler # type: ignore
  ORIENTATION_QUATERNION = mdda.XsPayloadMode_OrientationQuaternion # type: ignore
  ORIENTATION_EULER = mdda.XsPayloadMode_OrientationEuler # type: ignore
  FREE_ACCELERATION = mdda.XsPayloadMode_FreeAcceleration # type: ignore
  MFM = mdda.XsPayloadMode_MFM # type: ignore
  RATE_QUANTITIES_W_MAG = mdda.XsPayloadMode_RateQuantitieswMag # type: ignore
  RATE_QUANTITIES = mdda.XsPayloadMode_RateQuantities # type: ignore
  DELTA_QUANTITIES_W_MAG = mdda.XsPayloadMode_DeltaQuantitieswMag # type: ignore
  DELTA_QUANTITIES = mdda.XsPayloadMode_DeltaQuantities # type: ignore
  HIGH_FIDELITY_W_MAG = mdda.XsPayloadMode_HighFidelitywMag # type: ignore
  HIGH_FIDELITY = mdda.XsPayloadMode_HighFidelity # type: ignore
  CUSTOM_MODE1 = mdda.XsPayloadMode_CustomMode1 # type: ignore
  CUSTOM_MODE2 = mdda.XsPayloadMode_CustomMode2 # type: ignore
  CUSTOM_MODE3 = mdda.XsPayloadMode_CustomMode3 # type: ignore
  CUSTOM_MODE4 = mdda.XsPayloadMode_CustomMode4 # type: ignore
  CUSTOM_MODE5 = mdda.XsPayloadMode_CustomMode5 # type: ignore


MOVELLA_DATA_GET_METHODS = {
  "acceleration":         MovellaDataGetter(func=lambda packet: packet.calibratedAcceleration(),  n_dim=(3,), dtype=np.float32, type_str='float32'),
  "gyroscope":            MovellaDataGetter(func=lambda packet: packet.calibratedGyroscopeData(), n_dim=(3,), dtype=np.float32, type_str='float32'),
  "magnetometer":         MovellaDataGetter(func=lambda packet: packet.calibratedMagneticField(), n_dim=(3,), dtype=np.float32, type_str='float32'),
  "quaternion":           MovellaDataGetter(func=lambda packet: packet.orientationQuaternion(),   n_dim=(4,), dtype=np.float32, type_str='float32'),
  "euler":                MovellaDataGetter(func=lambda packet: packet.orientationEuler(),        n_dim=(3,), dtype=np.float32, type_str='float32'),
  "free_acceleration":    MovellaDataGetter(func=lambda packet: packet.freeAcceleration(),        n_dim=(3,), dtype=np.float32, type_str='float32'),
  "dq":                   MovellaDataGetter(func=lambda packet: packet.orientationIncrement(),    n_dim=(4,), dtype=np.float32, type_str='float32'),
  "dv":                   MovellaDataGetter(func=lambda packet: packet.velocityIncrement(),       n_dim=(3,), dtype=np.float32, type_str='float32'),
  "status":               MovellaDataGetter(func=lambda packet: packet.status(),                  n_dim=(),   dtype=np.uint16,  type_str='uint16'),
}

foo: Callable[[Iterable[str]], Mapping[str, MovellaDataGetter]] = lambda l: {k:v for k,v in MOVELLA_DATA_GET_METHODS.items() if k in l}

MOVELLA_PAYLOAD_MODE = {
  "ExtendedQuaternion":     MovellaPayloadTuple(num_bytes=36, payload_mode=MovellaPayloadEnum.EXTENDED_QUATERNION,    methods=foo(["quaternion",
                                                                                                                                   "free_acceleration",
                                                                                                                                   "status"])),
  "CompleteQuaternion":     MovellaPayloadTuple(num_bytes=32, payload_mode=MovellaPayloadEnum.COMPLETE_QUATERNION,    methods=foo(["quaternion",
                                                                                                                                   "free_acceleration"])),
  "ExtendedEuler":          MovellaPayloadTuple(num_bytes=32, payload_mode=MovellaPayloadEnum.EXTENDED_EULER,         methods=foo(["euler",
                                                                                                                                   "free_acceleration",
                                                                                                                                   "status"])),
  "CompleteEuler":          MovellaPayloadTuple(num_bytes=28, payload_mode=MovellaPayloadEnum.COMPLETE_EULER,         methods=foo(["quaternion",
                                                                                                                                   "free_acceleration"])),
  "OrientationQuaternion":  MovellaPayloadTuple(num_bytes=20, payload_mode=MovellaPayloadEnum.ORIENTATION_QUATERNION, methods=foo(["quaternion"])),
  "OrientationEuler":       MovellaPayloadTuple(num_bytes=16, payload_mode=MovellaPayloadEnum.ORIENTATION_EULER,      methods=foo(["euler"])),
  "FreeAcceleration":       MovellaPayloadTuple(num_bytes=16, payload_mode=MovellaPayloadEnum.FREE_ACCELERATION,      methods=foo(["free_acceleration"])),
  "MFM":                    MovellaPayloadTuple(num_bytes=16, payload_mode=MovellaPayloadEnum.MFM,                    methods=foo(["magnetometer"])),
  "RateQuantitieswMag":     MovellaPayloadTuple(num_bytes=34, payload_mode=MovellaPayloadEnum.RATE_QUANTITIES_W_MAG,  methods=foo(["acceleration",
                                                                                                                                   "gyroscope",
                                                                                                                                   "magnetometer"])),
  "RateQuantities":         MovellaPayloadTuple(num_bytes=28, payload_mode=MovellaPayloadEnum.RATE_QUANTITIES,        methods=foo(["acceleration",
                                                                                                                                   "gyroscope"])),
  "DeltaQuantitieswMag":    MovellaPayloadTuple(num_bytes=38, payload_mode=MovellaPayloadEnum.DELTA_QUANTITIES_W_MAG, methods=foo(["dq",
                                                                                                                                   "dv",
                                                                                                                                   "magnetometer"])),
  "DeltaQuantities":        MovellaPayloadTuple(num_bytes=32, payload_mode=MovellaPayloadEnum.DELTA_QUANTITIES,       methods=foo(["dq",
                                                                                                                                   "dv"])),
  "HighFidelitywMag":       MovellaPayloadTuple(num_bytes=35, payload_mode=MovellaPayloadEnum.HIGH_FIDELITY_W_MAG,    methods=foo(["acceleration",
                                                                                                                                   "gyroscope",
                                                                                                                                   "magnetometer"])),
  "HighFidelity":           MovellaPayloadTuple(num_bytes=29, payload_mode=MovellaPayloadEnum.HIGH_FIDELITY,          methods=foo(["acceleration",
                                                                                                                                   "gyroscope"])),
  "CustomMode1":            MovellaPayloadTuple(num_bytes=40, payload_mode=MovellaPayloadEnum.CUSTOM_MODE1,           methods=foo(["euler",
                                                                                                                                   "free_acceleration",
                                                                                                                                   "gyroscope"])),
  "CustomMode2":            MovellaPayloadTuple(num_bytes=34, payload_mode=MovellaPayloadEnum.CUSTOM_MODE2,           methods=foo(["euler",
                                                                                                                                   "free_acceleration",
                                                                                                                                   "magnetometer"])),
  "CustomMode3":            MovellaPayloadTuple(num_bytes=32, payload_mode=MovellaPayloadEnum.CUSTOM_MODE3,           methods=foo(["quaternion",
                                                                                                                                   "gyroscope"])),
  "CustomMode4":            MovellaPayloadTuple(num_bytes=51, payload_mode=MovellaPayloadEnum.CUSTOM_MODE4,           methods=foo(["quaternion",
                                                                                                                                   "acceleration",
                                                                                                                                   "gyroscope",
                                                                                                                                   "magnetometer",
                                                                                                                                   "status"])),
  "CustomMode5":            MovellaPayloadTuple(num_bytes=44, payload_mode=MovellaPayloadEnum.CUSTOM_MODE5,           methods=foo(["quaternion",
                                                                                                                                   "acceleration",
                                                                                                                                   "gyroscope"])),
}

MOVELLA_PAYLOAD_MODE["ExtendedQuaternion"]["methods"]
# NOTE: Movella sets different internal low-pass filter for different activities:
#         'General' - general human daily activities.
#         'Dynamic' - high-pace activities (e.g. sprints).
MOVELLA_LOGGING_MODE = {
  "Euler":        mdda.XsLogOptions_Euler, # type: ignore
  "Quaternion":   mdda.XsLogOptions_Quaternion # type: ignore
}
MOVELLA_STATUS_MASK = {
  0x0001: "Accelerometer out of range in x-axis",
  0x0002: "Accelerometer out of range in y-axis",
  0x0004: "Accelerometer out of range in z-axis",
  0x0008: "Gyroscope out of range in x-axis",
  0x0010: "Gyroscope out of range in y-axis",
  0x0020: "Gyroscope out of range in z-axis",
  0x0040: "Magnetometer out of range in x-axis",
  0x0080: "Magnetometer out of range in y-axis",
  0x0100: "Magnetometer out of range in z-axis",
}
