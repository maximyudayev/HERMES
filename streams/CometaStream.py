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
from visualizers import LinePlotVisualizer
from streams.Stream import Stream
import dash_bootstrap_components as dbc


###########################################
###########################################
# A structure to store Cometa EMG+IMU data.
###########################################
###########################################
class CometaStream(Stream):
  def __init__(self, 
               device_mapping: dict[str, str],
               sampling_rate_hz: int = 2000,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               **_) -> None:
    super().__init__()
    self._device_mapping = device_mapping
    self._sampling_rate_hz = sampling_rate_hz
    self._update_interval_ms = update_interval_ms
    self._num_sensors = len(device_mapping)
    self._timesteps_before_solidified = timesteps_before_solidified

    self._define_data_notes()

    self.add_stream(device_name='cometa-emg',
                    stream_name='emg',
                    data_type='float32',
                    sample_size=(self._num_sensors,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    is_measure_rate_hz=True,
                    data_notes=self._data_notes['cometa-emg']['emg'])
    self.add_stream(device_name='cometa-emg',
                    stream_name='num_samples',
                    data_type='int32',
                    sample_size=(1,),
                    data_notes=self._data_notes['cometa-emg']['num_samples'])
    self.add_stream(device_name='cometa-emg',
                    stream_name='toa_s',
                    data_type='float64',
                    sample_size=(1,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['cometa-emg']['toa_s'])


  def get_fps(self) -> dict[str, float | None]:
    return {'cometa-emg': super()._get_fps('cometa-emg', 'emg')}


  def build_visulizer(self) -> dbc.Row:
    electromyography_plot = LinePlotVisualizer(stream=self,
                                               unique_id='cometa_emg',
                                               data_path={'cometa-emg': ['emg']},
                                               legend_names=list(self._device_mapping.values()),
                                               plot_duration_timesteps=self._timesteps_before_solidified,
                                               update_interval_ms=self._update_interval_ms,
                                               col_width=6)
    return dbc.Row([electromyography_plot.layout])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('cometa-emg', {})

    self._data_notes['cometa-emg']['emg'] = OrderedDict([
      ('Description', 'Electric potential at the placement location'),
      ('Units', 'uV'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.keys())),
    ])
    self._data_notes['cometa-emg']['num_samples'] = OrderedDict([
      ('Description', 'Number of burst samples in the packet received at the current strobe.'),
    ])
    self._data_notes['cometa-emg']['toa_s'] = OrderedDict([
      ('Description', 'Time of arrival of the burst samples packet w.r.t. system clock.'),
      ('Units', 'seconds'),
    ])
