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
import dash_bootstrap_components as dbc
from streams import Stream
from visualizers import GazeVisualizer, VideoVisualizer


################################################
################################################
# A structure to store Pupil Core stream's data.
################################################
################################################
class EyeStream(Stream):
  def __init__(self,
               is_binocular: bool,
               is_stream_video_world: bool,
               is_stream_video_eye: bool,
               is_stream_fixation: bool,
               is_stream_blinks: bool,
               gaze_estimate_stale_s: float,
               shape_video_world: tuple,
               shape_video_eye0: tuple,
               shape_video_eye1: tuple,
               fps_video_world: float,
               fps_video_eye0: float,
               fps_video_eye1: float,
               pixel_format: str, # [bgr, yuv, jpeg]
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               **_) -> None:
    super().__init__()

    self._is_binocular = is_binocular
    self._is_stream_video_world = is_stream_video_world
    self._is_stream_video_eye = is_stream_video_eye
    self._is_stream_fixation = is_stream_fixation
    self._is_stream_blinks = is_stream_blinks
    self._gaze_estimate_stale_s = gaze_estimate_stale_s
    self._update_interval_ms = update_interval_ms
    self._pixel_format = pixel_format
    self._timesteps_before_solidified = timesteps_before_solidified

    # Define data notes that will be associated with streams created below.
    self._define_data_notes()

    # Create a stream for the Pupil Core time, to help evaluate drift and offsets.
    # Note that core time is included with each other stream as well,
    #  but include a dedicated one too just in case there are delays in sending
    #  the other data payloads.
    self.add_stream(device_name='eye-time', 
                    stream_name='device_time_s',
                    data_type='float64', 
                    sample_size=(1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-time']['device_time_s'])

    # Create streams for gaze data.
    self.add_stream(device_name='eye-gaze', 
                    stream_name='confidence',
                    data_type='float64', 
                    sample_size=(1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-gaze']['confidence'])
    self.add_stream(device_name='eye-gaze', 
                    stream_name='eye_center_3d',
                    data_type='float64', 
                    sample_size=(2,3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-gaze']['eye_center_3d'])
    self.add_stream(device_name='eye-gaze', 
                    stream_name='normal_3d',
                    data_type='float64', 
                    sample_size=(2,3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-gaze']['normal_3d'])
    self.add_stream(device_name='eye-gaze', 
                    stream_name='point_3d',
                    data_type='float64', 
                    sample_size=(3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-gaze']['point_3d'])
    self.add_stream(device_name='eye-gaze', 
                    stream_name='position',
                    data_type='float64', 
                    sample_size=(2),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-gaze']['position'])
    self.add_stream(device_name='eye-gaze', 
                    stream_name='timestamp',
                    data_type='float64', 
                    sample_size=(1),
                    sampling_rate_hz=fps_video_world,
                    is_measure_rate_hz=True,
                    data_notes=self._data_notes['eye-gaze']['timestamp'])

    # Create streams for pupil data.
    self.add_stream(device_name='eye-pupil', 
                    stream_name='confidence',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['confidence'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='circle3d_center',
                    data_type='float64', 
                    sample_size=(2,3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['circle3d_center'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='circle3d_normal',
                    data_type='float64', 
                    sample_size=(2,3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['circle3d_normal'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='circle3d_radius',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['circle3d_radius'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='diameter',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['diameter'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='diameter3d',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['diameter3d'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='polar_phi',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['polar_phi'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='polar_theta',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['polar_theta'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='position',
                    data_type='float64', 
                    sample_size=(2,2),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['position'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='projected_sphere_angle',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['projected_sphere_angle'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='projected_sphere_axes',
                    data_type='float64', 
                    sample_size=(2,2),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['projected_sphere_axes'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='projected_sphere_center',
                    data_type='float64', 
                    sample_size=(2,2),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['projected_sphere_center'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='sphere_center',
                    data_type='float64', 
                    sample_size=(2,3),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['sphere_center'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='sphere_radius',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    data_notes=self._data_notes['eye-pupil']['sphere_radius'])
    self.add_stream(device_name='eye-pupil', 
                    stream_name='timestamp',
                    data_type='float64', 
                    sample_size=(2,1),
                    sampling_rate_hz=fps_video_world,
                    is_measure_rate_hz=True,
                    data_notes=self._data_notes['eye-pupil']['timestamp'])

    # Create streams for fixation data.
    if is_stream_fixation:
      self.add_stream(device_name='eye-fixations',
                      stream_name='id',
                      data_type='int32',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['id'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='timestamp',
                      data_type='float64',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      is_measure_rate_hz=True,
                      data_notes=self._data_notes['eye-fixations']['timestamp'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='norm_pos',
                      data_type='float32',
                      sample_size=(2,1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['norm_pos'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='dispersion',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['dispersion'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='duration',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['duration'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='confidence',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['confidence'])
      self.add_stream(device_name='eye-fixations',
                      stream_name='gaze_point_3d',
                      data_type='float32',
                      sample_size=(3,1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-fixations']['gaze_point_3d'])
      
    # Create streams for blinks data.
    if is_stream_blinks:
      self.add_stream(device_name='eye-blinks',
                      stream_name='timestamp',
                      data_type='float64',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      is_measure_rate_hz=True,
                      data_notes=self._data_notes['eye-blinks']['timestamp'])
      self.add_stream(device_name='eye-blinks',
                      stream_name='confidence',
                      data_type='float32',
                      sample_size=(1),
                      sampling_rate_hz=0,
                      data_notes=self._data_notes['eye-blinks']['confidence'])
      

    # Create streams for video data.
    if is_stream_video_world:
      self.add_stream(device_name='eye-video-world',
                      stream_name='frame_timestamp',
                      data_type='float64',
                      sample_size=(1),
                      sampling_rate_hz=fps_video_world,
                      data_notes=self._data_notes['eye-video-world']['frame_timestamp'])
      self.add_stream(device_name='eye-video-world', 
                      stream_name='frame_index',
                      data_type='uint64', 
                      sample_size=(1),
                      sampling_rate_hz=fps_video_world, 
                      data_notes=self._data_notes['eye-video-world']['frame_index'])
      self.add_stream(device_name='eye-video-world', 
                      stream_name='frame',
                      data_type='uint8', 
                      sample_size=shape_video_world,
                      sampling_rate_hz=fps_video_world, 
                      data_notes=self._data_notes['eye-video-world']['frame'],
                      is_measure_rate_hz=True,
                      is_video=True,
                      color_format=self._pixel_format, # can be bgr, jpeg and yuv
                      timesteps_before_solidified=self._timesteps_before_solidified)

    if is_stream_video_eye:
      self.add_stream(device_name='eye-video-eye0', 
                      stream_name='frame_timestamp',
                      data_type='float64', 
                      sample_size=(1),
                      sampling_rate_hz=fps_video_eye0, 
                      data_notes=self._data_notes['eye-video-eye0']['frame_timestamp'])
      self.add_stream(device_name='eye-video-eye0', 
                      stream_name='frame_index',
                      data_type='uint64', 
                      sample_size=(1),
                      sampling_rate_hz=fps_video_eye0, 
                      data_notes=self._data_notes['eye-video-eye0']['frame_index'])
      self.add_stream(device_name='eye-video-eye0', 
                      stream_name='frame',
                      data_type='uint8', 
                      sample_size=shape_video_eye0,
                      sampling_rate_hz=fps_video_eye0, 
                      data_notes=self._data_notes['eye-video-eye0']['frame'],
                      is_measure_rate_hz=True,
                      is_video=True,
                      color_format=self._pixel_format,
                      timesteps_before_solidified=self._timesteps_before_solidified)
      if is_binocular:
        self.add_stream(device_name='eye-video-eye1', 
                        stream_name='frame_timestamp',
                        data_type='float64', 
                        sample_size=(1),
                        sampling_rate_hz=fps_video_eye1, 
                        data_notes=self._data_notes['eye-video-eye1']['frame_timestamp'])
        self.add_stream(device_name='eye-video-eye1', 
                        stream_name='frame_index',
                        data_type='uint64', 
                        sample_size=(1),
                        sampling_rate_hz=fps_video_eye1, 
                        data_notes=self._data_notes['eye-video-eye1']['frame_index'])
        self.add_stream(device_name='eye-video-eye1', 
                        stream_name='frame',
                        data_type='uint8', 
                        sample_size=shape_video_eye1,
                        sampling_rate_hz=fps_video_eye1, 
                        data_notes=self._data_notes['eye-video-eye1']['frame'],
                        is_measure_rate_hz=True,
                        is_video=True,
                        color_format=self._pixel_format,
                        timesteps_before_solidified=self._timesteps_before_solidified)


  def get_fps(self) -> dict[str, float]:
    fps = {
      'eye-gaze': super()._get_fps('eye-gaze', 'timestamp'),
      'eye-pupil': super()._get_fps('eye-pupil', 'timestamp'),
      'eye-fixations': super()._get_fps('eye-fixations', 'timestamp'),
      'eye-blinks': super()._get_fps('eye-blinks', 'timestamp'),
    }

    if self._is_stream_video_world:
      fps['eye-video-world'] = super()._get_fps('eye-video-world', 'frame')
    if self._is_stream_video_eye:
      fps['eye-video-eye0'] = super()._get_fps('eye-video-eye0', 'frame')
      if self._is_binocular:
        fps['eye-video-eye1'] = super()._get_fps('eye-video-eye1', 'frame')
    return fps


  def build_visulizer(self) -> dbc.Row:
    devices = ['eye-video-world',
               'eye-video-eye0',
               'eye-video-eye1']
    predicates = [self._is_stream_video_world,
                  self._is_stream_video_eye,
                  self._is_stream_video_eye and self._is_binocular]
    
    gaze_plot = [GazeVisualizer(stream=self,
                                unique_id=device,
                                world_data_path={'eye-video-world': 'frame'},
                                gaze_data_path={'eye-gaze': 'position'},
                                legend_name=device,
                                update_interval_ms=self._update_interval_ms,
                                col_width=6)
                    for device, predicate in zip(devices[0:1],predicates[0:1]) if predicate]

    eye_plots = [VideoVisualizer(stream=self,
                                 unique_id=device,
                                 data_path={device: 'frame'},
                                 legend_name=device,
                                 update_interval_ms=self._update_interval_ms,
                                 col_width=6)
                    for device, predicate in zip(devices[1:],predicates[1:]) if predicate]
    
    return dbc.Row([camera_plot.layout for camera_plot in gaze_plot+eye_plots])


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('eye-gaze', {})
    self._data_notes.setdefault('eye-pupil', {})
    self._data_notes.setdefault('eye-fixations', {})
    self._data_notes.setdefault('eye-blinks', {})
    self._data_notes.setdefault('eye-time', {})
    self._data_notes.setdefault('eye-video-eye0', {})
    self._data_notes.setdefault('eye-video-eye1', {})
    self._data_notes.setdefault('eye-video-world', {})

    # Gaze data
    self._data_notes['eye-gaze']['confidence'] = OrderedDict([
      ('Range', '[0, 1]'),
      ('Description', 'Confidence of the gaze detection'),
      ('PupilCapture key', 'gaze.Xd. > confidence'),
    ])
    self._data_notes['eye-gaze']['eye_center_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.3d. > eye_center_3d'),
    ])
    self._data_notes['eye-gaze']['normal_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.3d. > gaze_normal_3d'),
    ])
    self._data_notes['eye-gaze']['point_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.3d. > gaze_point_3d'),
    ])
    self._data_notes['eye-gaze']['position'] = OrderedDict([
      ('Description', 'The normalized gaze position in image space, corresponding to the world camera image'),
      ('Units', 'normalized between [0, 1]'),
      ('Origin', 'bottom left'),
      (Stream.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd. > norm_pos'),
    ])
    self._data_notes['eye-gaze']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'gaze.Xd. > timestamp'),
    ])

    # Pupil data
    self._data_notes['eye-pupil']['confidence'] = OrderedDict([
      ('Range', '[0, 1]'),
      ('Description', 'Confidence of the pupil detection'),
      ('PupilCapture key', 'gaze.Xd. > base_data > confidence'),
    ])
    self._data_notes['eye-pupil']['circle3d_center'] = OrderedDict([
      ('Units', 'mm'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.Xd. > base_data > circle_3d > center'),
    ])
    self._data_notes['eye-pupil']['circle3d_normal'] = OrderedDict([
      ('Units', 'mm'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.Xd. > base_data > circle_3d > normal'),
    ])
    self._data_notes['eye-pupil']['circle3d_radius'] = OrderedDict([
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd. > base_data > circle_3d > radius'),
    ])
    self._data_notes['eye-pupil']['diameter'] = OrderedDict([
      ('Units', 'pixels'),
      ('Notes', 'The estimated pupil diameter in image space, corresponding to the eye camera image'),
      ('PupilCapture key', 'gaze.Xd. > base_data > diameter'),
    ])
    self._data_notes['eye-pupil']['diameter3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'The estimated pupil diameter in 3D space'),
      ('PupilCapture key', 'gaze.Xd. > base_data > diameter_3d'),
    ])
    self._data_notes['eye-pupil']['polar_phi'] = OrderedDict([
      ('Notes', 'Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.'),
      ('See also', 'polar_theta is the other polar coordinate'),
      ('PupilCapture key', 'gaze.Xd. > base_data > phi'),
    ])
    self._data_notes['eye-pupil']['polar_theta'] = OrderedDict([
      ('Notes', 'Pupil polar coordinate on 3D eye model. The model assumes a fixed eye ball size, so there is no radius key.'),
      ('See also', 'polar_phi is the other polar coordinate'),
      ('PupilCapture key', 'gaze.Xd. > base_data > theta'),
    ])
    self._data_notes['eye-pupil']['position'] = OrderedDict([
      ('Description', 'The normalized pupil position in image space, corresponding to the eye camera image'),
      ('Units', 'normalized between [0, 1]'),
      ('Origin', 'bottom left'),
      (Stream.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd. > base_data > norm_pos'),
    ])
    self._data_notes['eye-pupil']['projected_sphere_angle'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'degrees'),
      ('PupilCapture key', 'gaze.Xd. > base_data > projected_sphere > angle'),
    ])
    self._data_notes['eye-pupil']['projected_sphere_axes'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'pixels'),
      ('Origin', 'bottom left'),
      ('PupilCapture key', 'gaze.Xd. > base_data > projected_sphere > axes'),
    ])
    self._data_notes['eye-pupil']['projected_sphere_center'] = OrderedDict([
      ('Description', 'Projection of the 3D eye ball sphere into image space corresponding to the eye camera image'),
      ('Units', 'pixels'),
      ('Origin', 'bottom left'),
      (Stream.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'gaze.Xd. > base_data > projected_sphere > center'),
    ])
    self._data_notes['eye-pupil']['sphere_center'] = OrderedDict([
      ('Description', 'The 3D eye ball sphere'),
      ('Units', 'mm'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'gaze.Xd. > base_data > sphere > center'),
    ])
    self._data_notes['eye-pupil']['sphere_radius'] = OrderedDict([
      ('Description', 'The 3D eye ball sphere'),
      ('Units', 'mm'),
      ('PupilCapture key', 'gaze.Xd. > base_data > sphere > radius'),
    ])
    self._data_notes['eye-pupil']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'gaze.Xd. > base_data > timestamp'),
    ])

    # Fixations data
    self._data_notes['eye-fixations']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'fixations. > timestamp'),
    ])
    self._data_notes['eye-fixations']['norm_pos'] = OrderedDict([
      ('Description', 'The normalized pupil position in image space, corresponding to the eye camera image'),
      ('Units', 'normalized between [0, 1]'),
      ('Origin', 'bottom left'),
      (Stream.metadata_data_headings_key, ['x','y']),
      ('PupilCapture key', 'fixations. > norm_pos'),
    ])
    self._data_notes['eye-fixations']['dispersion'] = OrderedDict([
      ('Range', ''),
      ('Description', ''),
      ('PupilCapture key', 'fixations. > dispersion'),
    ])
    self._data_notes['eye-fixations']['duration'] = OrderedDict([
      ('Range', ''),
      ('Description', ''),
      ('PupilCapture key', 'fixations. > duration'),
    ])
    self._data_notes['eye-fixations']['confidence'] = OrderedDict([
      ('Range', ''),
      ('Description', ''),
      ('PupilCapture key', 'fixations. > confidence'),
    ])
    self._data_notes['eye-fixations']['gaze_point_3d'] = OrderedDict([
      ('Units', 'mm'),
      ('Notes', 'Maps pupil positions into the world camera coordinate system'),
      (Stream.metadata_data_headings_key, ['x','y','z']),
      ('PupilCapture key', 'fixations. > gaze_point_3d'),
    ])

    # Blinks data
    self._data_notes['eye-blinks']['confidence'] = OrderedDict([
      ('Range', '[0, 1]'),
      ('Description', 'Confidence of the blink detection'),
      ('PupilCapture key', 'blinks. > confidence'),
    ])
    self._data_notes['eye-blinks']['timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Capture software, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ('PupilCapture key', 'blinks. > timestamp'),
    ])

    # Time
    self._data_notes['eye-time']['device_time_s'] = OrderedDict([
      ('Description', 'The timestamp fetched from the Pupil Core service, which can be used for alignment to system time in time_s.  '
                      'As soon as system time time_s was recorded, a command was sent to Pupil Capture to get its time; '
                      'so a slight communication delay is included on the order of milliseconds.  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])

    # Eye videos
    for i in range(2):
      self._data_notes['eye-video-eye%s' % i]['frame_timestamp'] = OrderedDict([
        ('Description', 'The timestamp recorded by the Pupil Core service, '
                        'which should be more precise than the system time when the data was received (the time_s field).  '
                        'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
      ])
      self._data_notes['eye-video-eye%s' % i]['frame_index'] = OrderedDict([
        ('Description', 'The frame index recorded by the Pupil Core service, '
                        'which relates to world frame used for annotation'),
      ])
      self._data_notes['eye-video-eye%s' % i]['frame'] = OrderedDict([
        ('Format', 'Frames are in BGR format'),
      ])
    # World video
    self._data_notes['eye-video-world']['frame_timestamp'] = OrderedDict([
      ('Description', 'The timestamp recorded by the Pupil Core service, '
                      'which should be more precise than the system time when the data was received (the time_s field).  '
                      'Note that Pupil Core time was synchronized with system time at the start of recording, accounting for communication delays.'),
    ])
    self._data_notes['eye-video-world']['frame_index'] = OrderedDict([
      ('Description', 'The frame index recorded by the Pupil Core service, '
                      'which relates to world frame used for annotation'),
    ])
    self._data_notes['eye-video-world']['frame'] = OrderedDict([
      ('Format', 'Frames are in BGR format'),
    ])
