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

from streams import Stream
import dash_bootstrap_components as dbc


###############################################
###############################################
# A structure to store TMSi SAGA stream's data.
###############################################
###############################################
class TmsiStream(Stream):
  def __init__(self, 
               sampling_rate_hz: int = 20,
               transmission_delay_period_s: int = None,
               **_) -> None:
    super().__init__()
    self._sampling_rate_hz = sampling_rate_hz
    self._transmission_delay_period_s = transmission_delay_period_s

    self.add_stream(device_name='tmsi-data',
                    stream_name='breath',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='tmsi-data',
                    stream_name='GSR',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='tmsi-data',
                    stream_name='SPO2',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='tmsi-data',
                    stream_name='BIP-01',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='tmsi-data',
                    stream_name='BIP-02',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='tmsi-data',
                    stream_name='counter',
                    data_type='int32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz,
                    is_measure_rate_hz=True)

    if self._transmission_delay_period_s:
      self.add_stream(device_name='tmsi-connection',
                      stream_name='transmission_delay',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=1.0/self._transmission_delay_period_s)


  def get_fps(self) -> dict[str, float]:
    return {'tmsi-data': super()._get_fps('tmsi-data', 'counter')}


  def build_visulizer(self) -> dbc.Row | None:
    return super().build_visulizer()
