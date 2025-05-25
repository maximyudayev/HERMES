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

import sys
import clr

from handlers.CometaWaveplus.CometaConstants import ImuAcqTypeEnum, FootSwProtocolEnum
from handlers.CometaWaveplus.CometaConstants import SamplingRateEnum
from handlers.CometaWaveplus.CometaFootSwTransducer import CometaFootSwTransducerEnabled, CometaFootSwTransducerThreshold

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from CyUSB import * # type: ignore


class CometaCaptureConfiguration(CaptureConfiguration): # type: ignore
  def get_sampling_rate(self) -> SamplingRateEnum:
    return self.get_SamplingRate()

  def set_sampling_rate(self, rate: SamplingRateEnum) -> None:
    self.set_SamplingRate(rate)

  def get_external_trigger_status(self) -> bool:
    return self.get_ExternalTriggerEnabled()

  def set_external_trigger_status(self, is_enabled: bool) -> None:
    self.set_ExternalTriggerEnabled(is_enabled)

  def get_trigger_level(self) -> int:
    return self.get_ExternalTriggerActiveLevel()

  def set_trigger_level(self, level: int) -> None:
    self.set_ExternalTriggerActiveLevel(level)

  def get_fsw_a_is_enabled(self) -> CometaFootSwTransducerEnabled:
    return self.get_FootSwATransducerEnabled()
  
  def set_fsw_a_is_enabled(self, fsw_transducer_enabled: CometaFootSwTransducerEnabled) -> None:
    self.set_FootSwATransducerEnabled(fsw_transducer_enabled)

  def get_fsw_a_threshold(self) -> CometaFootSwTransducerEnabled:
    return self.get_FootSwATransducerThreshold()
  
  def set_fsw_a_threshold(self, fsw_transducer_threshold: CometaFootSwTransducerThreshold) -> None:
    self.set_FootSwATransducerThreshold(fsw_transducer_threshold)

  def get_fsw_b_is_enabled(self) -> CometaFootSwTransducerEnabled:
    return self.get_FootSwBTransducerEnabled()

  def set_fsw_b_is_enabled(self, fsw_transducer_enabled: CometaFootSwTransducerEnabled) -> None:
    self.set_FootSwBTransducerEnabled(fsw_transducer_enabled)

  def get_fsw_b_threshold(self) -> CometaFootSwTransducerThreshold:
    return self.get_FootSwBTransducerThreshold()
  
  def set_fsw_b_threshold(self, fsw_transducer_threshold: CometaFootSwTransducerThreshold) -> None:
    self.set_FootSwBTransducerThreshold(fsw_transducer_threshold)

  def get_fsw_protocol(self) -> FootSwProtocolEnum:
    return self.get_FootSwProtocol()

  def set_fsw_protocol(self, protocol: FootSwProtocolEnum) -> None:
    self.set_FootSwProtocol(protocol)

  def get_imq_acq_type(self) -> ImuAcqTypeEnum:
    return self.get_IMU_AcqType()

  def set_imu_acq_type(self, acq_type: ImuAcqTypeEnum) -> None:
    self.set_IMU_AcqType(acq_type)
