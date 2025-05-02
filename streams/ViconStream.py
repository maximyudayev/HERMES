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

from collections import OrderedDict
from streams import Stream
import dash_bootstrap_components as dbc

from visualizers import LinePlotVisualizer


##########################################################
##########################################################
# A structure to store data streams from the Vicon system.
##########################################################
##########################################################
class ViconStream(Stream):
  def __init__(self,
               device_mapping: dict[str,str],
               sampling_rate_hz: int = 2000,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               **_) -> None:
    super().__init__()
    self._device_mapping = device_mapping
    self._num_devices = len(device_mapping)
    self._sampling_rate_hz = sampling_rate_hz
    self._timesteps_before_solidified = timesteps_before_solidified
    self._update_interval_ms = update_interval_ms

    self._define_data_notes()

    self.add_stream(device_name='vicon-data',
                    stream_name='emg',
                    data_type='float32',
                    sample_size=(self._num_devices),
                    sampling_rate_hz=sampling_rate_hz,
                    is_measure_rate_hz=True)
    self.add_stream(device_name='vicon-data',
                    stream_name='counter',
                    data_type='float32',
                    sample_size=(1),
                    sampling_rate_hz=sampling_rate_hz)
    self.add_stream(device_name='vicon-data',
                    stream_name='latency',
                    data_type='float32',
                    sample_size=(1),
                    sampling_rate_hz=sampling_rate_hz)


  def get_fps(self) -> dict[str, float]:
    return {'vicon-data': super()._get_fps('vicon-data', 'emg')}


  def build_visulizer(self) -> dbc.Row:
    emg_plot = LinePlotVisualizer(stream=self,
                                  data_path={'vicon-data': ['emg']},
                                  legend_names=list(self._device_mapping.keys()),
                                  plot_duration_timesteps=self._timesteps_before_solidified,
                                  update_interval_ms=self._update_interval_ms,
                                  col_width=6)
    return dbc.Row([emg_plot.layout])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('vicon-data', {})

    self._data_notes['vicon-data']['emg'] = OrderedDict([
      ('Description', 'Analog EMG measurements captured using the DAC of the Vicon system.'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.keys())),
    ])
    self._data_notes['vicon-data']['counter'] = OrderedDict([
      ('Description', 'Block frame number for a burst of EMG measurements, sent in 10ms bursts.'),
    ])
    self._data_notes['vicon-data']['latency'] = OrderedDict([
      ('Description', 'Transmission delay estimate from Vicon w.r.t. the on-sensor acquisition time.'),
    ])
