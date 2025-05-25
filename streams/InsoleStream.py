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
#from visualizers import InsolePressureVisualizer
import dash_bootstrap_components as dbc


####################################################
####################################################
# A structure to store Moticon Insole stream's data.
####################################################
####################################################
class InsoleStream(Stream):
  def __init__(self, 
               sampling_rate_hz: int = 100,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               transmission_delay_period_s: int | None = None,
               **_) -> None:
    super().__init__()
    self._sampling_rate_hz = sampling_rate_hz
    self._transmission_delay_period_s = transmission_delay_period_s
    self._timesteps_before_solidified = timesteps_before_solidified
    self._update_interval_ms = update_interval_ms

    self._define_data_notes()

    self.add_stream(device_name='insoles-data',
                    stream_name='toa_s',
                    data_type='float64',
                    sample_size=(1,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['insoles-data']['toa_s'])
    self.add_stream(device_name='insoles-data',
                    stream_name='timestamp',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz,
                    is_measure_rate_hz=True)
    self.add_stream(device_name='insoles-data',
                    stream_name='foot_pressure_left',
                    data_type='float32',
                    sample_size=[16],
                    sampling_rate_hz=self._sampling_rate_hz,
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='insoles-data',
                    stream_name='foot_pressure_right',
                    data_type='float32',
                    sample_size=[16],
                    sampling_rate_hz=self._sampling_rate_hz,
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='insoles-data',
                    stream_name='acc_left',
                    data_type='float32',
                    sample_size=[3],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='acc_right',
                    data_type='float32',
                    sample_size=[3],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='gyro_left',
                    data_type='float32',
                    sample_size=[3],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='gyro_right',
                    data_type='float32',
                    sample_size=[3],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='total_force_left',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='total_force_right',
                    data_type='float32',
                    sample_size=[1],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='center_of_pressure_left',
                    data_type='float32',
                    sample_size=[2],
                    sampling_rate_hz=self._sampling_rate_hz)
    self.add_stream(device_name='insoles-data',
                    stream_name='center_of_pressure_right',
                    data_type='float32',
                    sample_size=[2],
                    sampling_rate_hz=self._sampling_rate_hz)
    
    if self._transmission_delay_period_s:
      self.add_stream(device_name='insoles-connection',
                      stream_name='transmission_delay',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=1.0/self._transmission_delay_period_s,
                      data_notes=self._data_notes['insoles-connection']['transmission_delay'])


  def get_fps(self) -> dict[str, float | None]:
    return {'insoles-data': super()._get_fps('insoles-data', 'timestamp')}


  # Visualize the foot pressure data from the 16 sensors per side.
  # https://moticon.com/wp-content/uploads/2021/09/OpenGo-Sensor-Insole-Specification-A4-RGB-EN-03.03.pdf (p.4)
  def build_visulizer(self) -> dbc.Row | None:
    return super().build_visulizer()
  # def build_visulizer(self) -> dbc.Row | None:
  #   insole_pressure_plot = InsolePressureVisualizer(stream=self,
  #                                                   device_name='insoles-data',
  #                                                   stream_names=['foot_pressure_left',
  #                                                                 'foot_pressure_right'],
  #                                                   legend_names=['pressure'],
  #                                                   plot_duration_timesteps=self._timesteps_before_solidified,
  #                                                   update_interval_ms=self._update_interval_ms,
  #                                                   col_width=6)
  #   return dbc.Row([insole_pressure_plot.layout])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('insoles-data', {})
    self._data_notes.setdefault('insoles-connection', {})

    self._data_notes['insoles-data']['timestamp'] = OrderedDict([
      ('Description', 'Device time of sampling of the insole data'),
    ])
    self._data_notes['insoles-data']['toa_s'] = OrderedDict([
      ('Description', 'Time of arrival of the packet w.r.t. system clock.'),
      ('Units', 'seconds'),
    ])
    self._data_notes['insoles-data']['foot_pressure_left'] = OrderedDict([
      ('Description', 'Pressure across the 16 strain gauge grid across the left insole'),
    ])
    self._data_notes['insoles-data']['foot_pressure_right'] = OrderedDict([
      ('Description', 'Pressure across the 16 strain gauge grid across the right insole'),
    ])
    self._data_notes['insoles-data']['acc_left'] = OrderedDict([
      ('Description', 'Acceleration in the X direction'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['insoles-data']['acc_right'] = OrderedDict([
      ('Description', 'Acceleration in the X direction'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['insoles-data']['gyro_left'] = OrderedDict([
      ('Description', 'Acceleration in the X direction'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['insoles-data']['gyro_right'] = OrderedDict([
      ('Description', 'Acceleration in the X direction'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
    ])
    self._data_notes['insoles-data']['total_force_left'] = OrderedDict([
      ('Description', 'Total force on the left insole'),
    ])
    self._data_notes['insoles-data']['total_force_right'] = OrderedDict([
      ('Description', 'Total force on the right insole'),
    ])
    self._data_notes['insoles-data']['center_of_pressure_left'] = OrderedDict([
      ('Description', 'Point of pressure concentration on the left insole'),
      (Stream.metadata_data_headings_key, ['x','y']),
    ])
    self._data_notes['insoles-data']['center_of_pressure_right'] = OrderedDict([
      ('Description', 'Point of pressure concentration on the right insole'),
      (Stream.metadata_data_headings_key, ['x','y']),
    ])
    self._data_notes['insoles-connection']['transmission_delay'] = OrderedDict([
      ('Description', 'Periodic transmission delay estimate of the connection link to the sensor'),
      ('Units', 'seconds'),
      ('Sample period', self._transmission_delay_period_s),
    ])
