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
from visualizers import LinePlotVisualizer#, SkeletonVisualizer
from streams.Stream import Stream
import dash_bootstrap_components as dbc


##########################################
##########################################
# A structure to store DOTs stream's data.
##########################################
##########################################
class DotsStream(Stream):
  def __init__(self, 
               device_mapping: dict[str,str],
               num_joints: int = 5,
               sampling_rate_hz: int = 60,
               is_get_orientation: bool = True,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               transmission_delay_period_s: int = None,
               **_) -> None:
    super().__init__()
    self._num_joints = num_joints
    self._sampling_rate_hz = sampling_rate_hz
    self._is_get_orientation = is_get_orientation
    self._transmission_delay_period_s = transmission_delay_period_s
    self._timesteps_before_solidified = timesteps_before_solidified
    self._update_interval_ms = update_interval_ms

    # Invert device mapping to map device_id -> joint_name
    (joint_names, device_ids) = tuple(zip(*(device_mapping.items())))
    self._device_mapping: OrderedDict[str, str] = OrderedDict(zip(device_ids, joint_names))

    self._define_data_notes()

    self.add_stream(device_name='dots-imu',
                    stream_name='acceleration-x',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['acceleration-x'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='acceleration-y',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['acceleration-y'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='acceleration-z',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['acceleration-z'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='gyroscope-x',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['gyroscope-x'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='gyroscope-y',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['gyroscope-y'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='gyroscope-z',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['gyroscope-z'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='magnetometer-x',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['magnetometer-x'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='magnetometer-y',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['magnetometer-y'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='dots-imu',
                    stream_name='magnetometer-z',
                    data_type='float32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['magnetometer-z'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    if self._is_get_orientation:
      self.add_stream(device_name='dots-imu',
                      stream_name='quaternion',
                      data_type='float32',
                      sample_size=(self._num_joints, 4),
                      sampling_rate_hz=self._sampling_rate_hz, 
                      data_notes=self._data_notes['dots-imu']['quaternion'])
    self.add_stream(device_name='dots-imu',
                    stream_name='timestamp',
                    data_type='uint32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    is_measure_rate_hz=True,
                    data_notes=self._data_notes['dots-imu']['timestamp'])
    self.add_stream(device_name='dots-imu',
                    stream_name='toa_s',
                    data_type='float64',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['toa_s'])
    self.add_stream(device_name='dots-imu',
                    stream_name='counter',
                    data_type='uint32',
                    sample_size=(self._num_joints),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['dots-imu']['counter'])

    if self._transmission_delay_period_s:
      self.add_stream(device_name='dots-connection',
                      stream_name='transmission_delay',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=1.0/self._transmission_delay_period_s,
                      data_notes=self._data_notes['dots-connection']['transmission_delay'])


  def get_fps(self) -> dict[str, float]:
    return {'dots-imu': super()._get_fps('dots-imu', 'timestamp')}


  # TODO: add `SkeletonVisualizer` for orientation data.
  def build_visulizer(self) -> dbc.Row:
    acceleration_plot = LinePlotVisualizer(stream=self,
                                           unique_id='dots_acc',
                                           data_path={'dots-imu': [
                                                        'acceleration-x',
                                                        'acceleration-y',
                                                        'acceleration-z']},
                                           legend_names=list(self._device_mapping.values()),
                                           plot_duration_timesteps=self._timesteps_before_solidified,
                                           update_interval_ms=self._update_interval_ms,
                                           col_width=6)
    gyroscope_plot = LinePlotVisualizer(stream=self,
                                        unique_id='dots_gyr',
                                        data_path={'dots-imu': [
                                                      'gyroscope-x',
                                                      'gyroscope-y',
                                                      'gyroscope-z']},
                                        legend_names=list(self._device_mapping.values()),
                                        plot_duration_timesteps=self._timesteps_before_solidified,
                                        update_interval_ms=self._update_interval_ms,
                                        col_width=6)
    magnetometer_plot = LinePlotVisualizer(stream=self,
                                           unique_id='dots_mag',
                                           data_path={'dots-imu': [
                                                        'magnetometer-x',
                                                        'magnetometer-y',
                                                        'magnetometer-z']},
                                           legend_names=list(self._device_mapping.values()),
                                           plot_duration_timesteps=self._timesteps_before_solidified,
                                           update_interval_ms=self._update_interval_ms,
                                           col_width=6)
    # skeleton_plot = SkeletonVisualizer()
    return dbc.Row([acceleration_plot.layout, gyroscope_plot.layout, magnetometer_plot.layout])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('dots-imu', {})
    self._data_notes.setdefault('dots-connection', {})

    self._data_notes['dots-imu']['acceleration-x'] = OrderedDict([
      ('Description', 'Linear acceleration in the X direction w.r.t. sensor local coordinate system'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['acceleration-y'] = OrderedDict([
      ('Description', 'Linear acceleration in the Y direction w.r.t. sensor local coordinate system'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['acceleration-z'] = OrderedDict([
      ('Description', 'Linear acceleration in the Z direction w.r.t. sensor local coordinate system'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['gyroscope-x'] = OrderedDict([
      ('Description', 'Angular velocity in the X direction w.r.t. sensor local coordinate system'),
      ('Units', 'degree/second'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['gyroscope-y'] = OrderedDict([
      ('Description', 'Angular velocity in the Y direction w.r.t. sensor local coordinate system'),
      ('Units', 'degree/second'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['gyroscope-z'] = OrderedDict([
      ('Description', 'Angular velocity in the Z direction w.r.t. sensor local coordinate system'),
      ('Units', 'degree/second'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['magnetometer-x'] = OrderedDict([
      ('Description', 'Magnetometer reading in the X direction'),
      ('Units', 'arbitrary unit normalized to earth field strength during factory calibration (~40uT), '
                'w.r.t. sensor local coordinate system'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['magnetometer-y'] = OrderedDict([
      ('Description', 'Magnetometer reading in the Y direction'),
      ('Units', 'arbitrary unit normalized to earth field strength during factory calibration (~40uT), '
                'w.r.t. sensor local coordinate system'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['magnetometer-z'] = OrderedDict([
      ('Description', 'Magnetometer reading in the Z direction'),
      ('Units', 'arbitrary unit normalized to earth field strength during factory calibration (~40uT), '
                'w.r.t. sensor local coordinate system'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    if self._is_get_orientation:
      self._data_notes['dots-imu']['quaternion'] = OrderedDict([
        ('Description', 'Quaternion rotation vector [W,X,Y,Z]'),
        (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
      ])
    self._data_notes['dots-imu']['timestamp'] = OrderedDict([
      ('Description', 'Time of sampling of the packet w.r.t. sensor on-board 1MHz clock, '
                      'clearing on startup and overflowing every ~1.2 hours'),
      ('Units', 'microsecond in range [0, (2^32)-1]'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['toa_s'] = OrderedDict([
      ('Description', 'Time of arrival of the packet w.r.t. system clock.'),
      ('Units', 'seconds'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-imu']['counter'] = OrderedDict([
      ('Description', 'Index of the sampled packet per device, w.r.t. the start of the recording, starting from 0. '
                      'At sample rate of 60Hz, corresponds to ~19884 hours of recording, longer than the battery life of the sensors.'),
      ('Range', '[0, (2^32)-1]'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['dots-connection']['transmission_delay'] = OrderedDict([
      ('Description', 'Periodic transmission delay estimate of the connection link to the sensor, '
                      'inter-tracker synchronization characterized by Movella under 10 microseconds'),
      ('Units', 'seconds'),
      ('Sample period', self._transmission_delay_period_s),
    ])
