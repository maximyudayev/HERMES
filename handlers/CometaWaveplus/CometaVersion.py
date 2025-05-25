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

sys.path.append("handlers/CometaWaveplus")
clr.AddReference("Waveplus.DaqSys") # type: ignore
clr.AddReference("Waveplus.DaqSysInterface") # type: ignore
clr.AddReference("CyUSB") # type: ignore

from Waveplus.DaqSys import * # type: ignore
from Waveplus.DaqSysInterface import * # type: ignore
from Waveplus.DaqSys.Definitions import * # type: ignore
from Waveplus.DaqSys.Exceptions import * # type: ignore
from CyUSB import * # type: ignore


class CometaVersion(Version): # type: ignore
  def get_major(self) -> int:
    return self.get_Major()
  
  def get_minor(self) -> int:
    return self.get_Minor()


class CometaExtVersion(ExtVersion): # type: ignore
  def get_major(self) -> int:
    return self.get_Major()
    
  def get_minor(self) -> int:
    return self.get_Minor()
    
  def get_build(self) -> int:
    return self.get_Build()
    
  def get_revision(self) -> int:
    return self.get_Revision()
