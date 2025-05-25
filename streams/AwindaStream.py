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
import dash_bootstrap_components as dbc


##################################################
##################################################
# A structure to store Awinda MTws' stream's data.
#   10-15 seconds after putting devices into
#   measurement mode is needed for orientation
#   computation to stabilize.
##################################################
##################################################
class AwindaStream(Stream):
  def __init__(self, 
               device_mapping: dict[str, str],
               num_joints: int = 7,
               sampling_rate_hz: int = 100,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               transmission_delay_period_s: int | None = None,
               **_) -> None:
    super().__init__()
    
    self._num_joints = num_joints
    self._sampling_rate_hz = sampling_rate_hz
    self._transmission_delay_period_s = transmission_delay_period_s
    self._timesteps_before_solidified = timesteps_before_solidified
    self._update_interval_ms = update_interval_ms
    
    # Invert device mapping to map device_id -> joint_name
    joint_names, device_ids = tuple(zip(*(device_mapping.items())))
    self._device_mapping: OrderedDict[str, str] = OrderedDict(zip(device_ids, joint_names))

    self._define_data_notes()

    # When using onLiveDataAvailable, every immediately available packet from each MTw is pushed in its own corresponding Stream.
    # When using onAllLiveDataAvailable, packets are packaged all at once (potentially for multiple timesteps)
    #   with interpolation of data for steps where some of sensors missed a measurement.
    # Choose the desired behavior for the system later. (currently onAllLiveDataAvailable).
    self.add_stream(device_name='awinda-imu',
                    stream_name='acceleration',
                    data_type='float32',
                    sample_size=(self._num_joints, 3),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['acceleration'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='awinda-imu',
                    stream_name='gyroscope',
                    data_type='float32',
                    sample_size=(self._num_joints, 3),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['gyroscope'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='awinda-imu',
                    stream_name='magnetometer',
                    data_type='float32',
                    sample_size=(self._num_joints, 3),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['magnetometer'],
                    timesteps_before_solidified=self._timesteps_before_solidified)
    self.add_stream(device_name='awinda-imu',
                    stream_name='quaternion',
                    data_type='float32',
                    sample_size=(self._num_joints, 4),
                    sampling_rate_hz=self._sampling_rate_hz, 
                    data_notes=self._data_notes['awinda-imu']['quaternion'])
    self.add_stream(device_name='awinda-imu',
                    stream_name='timestamp',
                    data_type='uint32',
                    sample_size=(self._num_joints,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    is_measure_rate_hz=True, # only 1 stream per device needs to be marked `True` if all streams get new data at a time
                    data_notes=self._data_notes['awinda-imu']['timestamp'])
    self.add_stream(device_name='awinda-imu',
                    stream_name='toa_s',
                    data_type='float64',
                    sample_size=(self._num_joints,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['toa_s'])
    self.add_stream(device_name='awinda-imu',
                    stream_name='counter_onboard',
                    data_type='uint16',
                    sample_size=(self._num_joints,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['counter_onboard'])
    self.add_stream(device_name='awinda-imu',
                    stream_name='counter',
                    data_type='uint32',
                    sample_size=(self._num_joints,),
                    sampling_rate_hz=self._sampling_rate_hz,
                    data_notes=self._data_notes['awinda-imu']['counter'])

    if self._transmission_delay_period_s:
      self.add_stream(device_name='awinda-connection',
                      stream_name='transmission_delay',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=1.0/self._transmission_delay_period_s,
                      data_notes=self._data_notes['awinda-connection']['transmission_delay'])


  def get_fps(self) -> dict[str, float | None]:
    return {'awinda-imu': super()._get_fps('awinda-imu', 'timestamp')}


  # TODO: add `SkeletonVisualizer` for orientation data.
  def build_visulizer(self) -> dbc.Row:
    acceleration_plot = LinePlotVisualizer(unique_id='awinda-acc',
                                           stream=self,
                                           data_path={'awinda-imu': ['acceleration']},
                                           legend_names=list(self._device_mapping.values()),
                                           plot_duration_timesteps=self._timesteps_before_solidified,
                                           update_interval_ms=self._update_interval_ms,
                                           col_width=6)
    gyroscope_plot = LinePlotVisualizer(unique_id='awinda-gyr',
                                        stream=self,
                                        data_path={'awinda-imu': ['gyroscope']},
                                        legend_names=list(self._device_mapping.values()),
                                        plot_duration_timesteps=self._timesteps_before_solidified,
                                        update_interval_ms=self._update_interval_ms,
                                        col_width=6)
    magnetometer_plot = LinePlotVisualizer(unique_id='awinda-mag',
                                           stream=self,
                                           data_path={'awinda-imu': ['magnetometer']},
                                           legend_names=list(self._device_mapping.values()),
                                           plot_duration_timesteps=self._timesteps_before_solidified,
                                           update_interval_ms=self._update_interval_ms,
                                           col_width=6)
    # skeleton_plot = SkeletonVisualizer()
    return dbc.Row([acceleration_plot.layout, gyroscope_plot.layout, magnetometer_plot.layout])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('awinda-imu', {})
    self._data_notes.setdefault('awinda-connection', {})

    self._data_notes['awinda-imu']['acceleration'] = OrderedDict([
      ('Description', 'Linear acceleration vector [X,Y,Z] w.r.t. sensor local coordinate system, '
                      'from SDI, integrated values converted to calibrated sensor data'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['gyroscope'] = OrderedDict([
      ('Description', 'Angular velocity vector [X,Y,Z] w.r.t. sensor local coordinate system, '
                      'from SDI, integrated values converted to calibrated sensor data'),
      ('Units', 'rad/second'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['magnetometer'] = OrderedDict([
      ('Description', 'Magnetic field  vector [X,Y,Z] w.r.t. sensor local coordinate system, '
                      'from SDI, integrated values converted to calibrated sensor data'),
      ('Units', 'arbitrary unit normalized to earth field strength during factory calibration, '
                'w.r.t. sensor local coordinate system'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['quaternion'] = OrderedDict([
      ('Description', 'Quaternion rotation vector [W,X,Y,Z]'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['timestamp'] = OrderedDict([
      ('Description', 'Time of sampling of the packet w.r.t. sensor on-board 1MHz clock, '
                      'clearing on startup and overflowing every ~1.2 hours'),
      ('Units', 'microsecond in range [0, (2^32)-1]'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['toa_s'] = OrderedDict([
      ('Description', 'Time of arrival of the packet w.r.t. system clock.'),
      ('Units', 'seconds'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['counter'] = OrderedDict([
      ('Description', 'Index of the sampled packet per device, w.r.t. the start of the recording, starting from 0. '
                      'At sample rate of 60Hz, corresponds to ~19884 hours of recording, longer than the battery life of the sensors.'),
      ('Range', '[0, (2^32)-1]'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-imu']['counter_onboard'] = OrderedDict([
      ('Description', 'Index of the sampled packet per device, starting from 0 on 1st read-out and wrapping around after 65535'),
      (Stream.metadata_data_headings_key, list(self._device_mapping.values())),
    ])
    self._data_notes['awinda-connection']['transmission_delay'] = OrderedDict([
      ('Description', 'Periodic transmission delay estimate of the connection link to the sensor, '
                      'inter-tracker synchronization characterized by Xsens under 10 microseconds'),
      ('Units', 'seconds'),
      ('Sample period', self._transmission_delay_period_s),
    ])
