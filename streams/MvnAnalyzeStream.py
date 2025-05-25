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


from collections import OrderedDict, namedtuple
from enum import Enum
from streams import Stream
from visualizers import LinePlotVisualizer#, SkeletonVisualizer
from streams.Stream import Stream
import dash_bootstrap_components as dbc


MVN_SEGMENT_MAPPING = {
  1: 'Pelvis',
  2: 'L5',
  3: 'L3',
  4: 'T12',
  5: 'T8',
  6: 'Neck',
  7: 'Head',
  8: 'Right Shoulder',
  9: 'Right Upper Arm',
  10: 'Right Forearm',
  11: 'Right Hand',
  12: 'Left Shoulder',
  13: 'Left Upper Arm',
  14: 'Left Forearm',
  15: 'Left Hand',
  16: 'Right Upper Leg',
  17: 'Right Lower Leg',
  18: 'Right Foot',
  19: 'Right Toe',
  20: 'Left Upper Leg',
  21: 'Left Lower Leg',
  22: 'Left Foot',
  23: 'Left Toe',
}

MVN_SENSOR_MAPPING = {
  1: 'Pelvis',
  5: 'T8',
  7: 'Head',
  8: 'Right Shoulder',
  9: 'Right Upper Arm',
  10: 'Right Forearm',
  11: 'Right Hand',
  12: 'Left Shoulder',
  13: 'Left Upper Arm',
  14: 'Left Forearm',
  15: 'Left Hand',
  16: 'Right Upper Leg',
  17: 'Right Lower Leg',
  18: 'Right Foot',
  20: 'Left Upper Leg',
  21: 'Left Lower Leg',
  22: 'Left Foot',
}

MvnJointDetails = namedtuple('MvnJointDetails', ['joint_id', 'name', 'description'])

# NOTE: magic key is the `parent-point_id` << 8 + `child_point_id`, used as hash key in the LUT.
MVN_JOINT_MAPPING = {
  0x01020201: MvnJointDetails(1,  'L5S1',             'Joint between the lumbar spine segment 5 and sacral spine 1 (ZXY)'),
  0x02020301: MvnJointDetails(2,  'L4L3',             'Joint between the lumbar spine segment 4 and lumbar spine segment 3 (ZXY)'),
  0x03020401: MvnJointDetails(3,  'L1T12',            'Joint between the lumbar spine segment 1 and thoracic spine segment 12 (ZXY)'),
  0x04020501: MvnJointDetails(4,  'T9T8',             ''),
  0x05020601: MvnJointDetails(5,  'T1C7',             ''),
  0x06020701: MvnJointDetails(6,  'C1Head',           'Joint between the cervical spine 1 and the head segment (ZXY)'),
  0x05030801: MvnJointDetails(7,  'RightT4Shoulder',  'Joint between thoracic spine 7 and the MVN shoulder segment'),
  0x08020901: MvnJointDetails(8,  'RightShoulder',    'Shoulder joint angle between the MVN shoulder segment and the upper arm; calculated using the Euler sequence ZXY. '
                                                      'Shoulder joint angle between the MVN shoulder segment and the upper arm; calculated using the Euler sequence XZY'),
  0x09020A01: MvnJointDetails(9,  'RightElbow',       'Joint between the upper arm and the forearm. (ZXY)'),
  0x0A020B01: MvnJointDetails(10, 'RightWrist',       'Joint between the forearm and the hand. (ZXY)'),
  0x05040C01: MvnJointDetails(11, 'LeftT4Shoulder',   'Joint between thoracic spine 7 and the MVN shoulder segment'),
  0x0C020D01: MvnJointDetails(12, 'LeftShoulder',     'Shoulder joint angle between the MVN shoulder segment and the upper arm; calculated using the Euler sequence ZXY. '
                                                      'Shoulder joint angle between the MVN shoulder segment and the upper arm; calculated using the Euler sequence XZY'),
  0x0D020E01: MvnJointDetails(13, 'LeftElbow',        'Joint between the upper arm and the forearm. (ZXY)'),
  0x0E020F01: MvnJointDetails(14, 'LeftWrist',        'Joint between the forearm and the hand. (ZXY)'),
  0x01031001: MvnJointDetails(15, 'RightHip',         'Joint between the pelvis and upper leg. (ZXY)'),
  0x10021101: MvnJointDetails(16, 'RightKnee',        'Joint between the upper leg and lower leg. (ZXY)'),
  0x11021201: MvnJointDetails(17, 'RightAnkle',       'Joint between the lower leg and foot. (ZXY)'),
  0x12021301: MvnJointDetails(18, 'RightBallFoot',    'Joint between the foot and the calculated toe. (ZXY)'),
  0x01041401: MvnJointDetails(19, 'LeftHip',          'Joint between the pelvis and upper leg. (ZXY)'),
  0x14021501: MvnJointDetails(20, 'LeftKnee',         'Joint between the upper leg and lower leg. (ZXY)'),
  0x15021601: MvnJointDetails(21, 'LeftAnkle',        'Joint between the lower leg and foot. (ZXY)'),
  0x16021701: MvnJointDetails(22, 'LeftBallFoot',     'Joint between the foot and the calculated toe. (ZXY)'),
  0x05000700: MvnJointDetails(23, 'T8Head',           'Ergonomic joint between the sternum and the head. (ZXY)'),
  0x05000D00: MvnJointDetails(24, 'T8LeftUpperArm',   'Ergonomic joint between sternum and the left shoulder level. (ZXY)'),
  0x05000900: MvnJointDetails(25, 'T8RightUpperArm',  'Ergonomic joint between sternum and the right shoulder level. (ZXY)'),
  0x01000500: MvnJointDetails(26, 'PelvisT8',         'Ergonomic joint between the pelvis and the sternum. (ZXY)'),
  0x01000100: MvnJointDetails(27, 'VerticalPelvis',   'Ergonomic joint describing pelvis tilt w.r.t global coordinate system. (ZXY)'),
  0x01000500: MvnJointDetails(28, 'VerticalT8',       'Ergonomic joint describing chest till w.r.t global coordinate system. (ZXY)'),
}

class MvnSegmentSetup(dict, Enum):
  FULL_BODY = MVN_SEGMENT_MAPPING
  FULL_BODY_NO_HANDS = {k:v for k, v in MVN_SEGMENT_MAPPING.items() if k not in [11,15]}
  LOWER_BODY = {k:v for k, v in MVN_SEGMENT_MAPPING.items() if k in [1,2,3,15,16,17,18,19,20,21,22]}
  LOWER_BODY_W_STERNUM = {k:v for k, v in MVN_SEGMENT_MAPPING.items() if k in [1,2,3,4,5,12,13,14,15,16,17]}
  UPPER_BODY = {k:v for k, v in MVN_SEGMENT_MAPPING.items() if k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
  UPPER_BODY_NO_HANDS = {k:v for k, v in MVN_SEGMENT_MAPPING.items() if k in [1,2,3,4,5,6,7,8,9,10,12,13,14]}

class MvnJointSetup(dict, Enum):
  FULL_BODY = MVN_JOINT_MAPPING
  FULL_BODY_NO_HANDS = {k:v for k, v in MVN_JOINT_MAPPING.items() if v.joint_id not in [10,14]}
  LOWER_BODY = {k:v for k, v in MVN_JOINT_MAPPING.items() if v.joint_id in [15,16,17,18,19,20,21,22,27]}
  LOWER_BODY_W_STERNUM = {k:v for k, v in MVN_JOINT_MAPPING.items() if v.joint_id in [1,2,3,4,15,16,17,18,19,20,21,22,26,27,28]}
  UPPER_BODY = {k:v for k, v in MVN_JOINT_MAPPING.items() if v.joint_id in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,23,24,25,26,27,28]}
  UPPER_BODY_NO_HANDS = {k:v for k, v in MVN_JOINT_MAPPING.items() if v.joint_id in [1,2,3,4,5,6,7,8,9,11,12,13,23,24,25,26,27,28]}

class MvnSensorSetup(dict, Enum):
  FULL_BODY = MVN_SENSOR_MAPPING
  FULL_BODY_NO_HANDS = {k:v for k, v in MVN_SENSOR_MAPPING.items() if k not in [7,11]}
  LOWER_BODY = {k:v for k, v in MVN_SENSOR_MAPPING.items() if k in [1,12,13,14,15,16,17]}
  LOWER_BODY_W_STERNUM = {k:v for k, v in MVN_SENSOR_MAPPING.items() if k in [1,2,12,13,14,15,16,17]}
  UPPER_BODY = {k:v for k, v in MVN_SENSOR_MAPPING.items() if k in [1,2,3,4,5,6,7,8,9,10,11]}
  UPPER_BODY_NO_HANDS = {k:v for k, v in MVN_SENSOR_MAPPING.items() if k in [1,2,3,4,5,6,8,9,10]}


MVN_SEGMENT_SETUP = {
  "full_body":            MvnSegmentSetup.FULL_BODY,
  "full_body_no_hands":   MvnSegmentSetup.FULL_BODY_NO_HANDS,
  "lower_body":           MvnSegmentSetup.LOWER_BODY,
  "lower_body_w_sternum": MvnSegmentSetup.LOWER_BODY_W_STERNUM,
  "upper_body":           MvnSegmentSetup.UPPER_BODY,
  "upper_body_no_hands":  MvnSegmentSetup.UPPER_BODY_NO_HANDS,
}

MVN_JOINT_SETUP = {
  "full_body":            MvnJointSetup.FULL_BODY,
  "full_body_no_hands":   MvnJointSetup.FULL_BODY_NO_HANDS,
  "lower_body":           MvnJointSetup.LOWER_BODY,
  "lower_body_w_sternum": MvnJointSetup.LOWER_BODY_W_STERNUM,
  "upper_body":           MvnJointSetup.UPPER_BODY,
  "upper_body_no_hands":  MvnJointSetup.UPPER_BODY_NO_HANDS,
}

MVN_SENSOR_SETUP = {
  "full_body":            MvnSensorSetup.FULL_BODY,
  "full_body_no_hands":   MvnSensorSetup.FULL_BODY_NO_HANDS,
  "lower_body":           MvnSensorSetup.LOWER_BODY,
  "lower_body_w_sternum": MvnSensorSetup.LOWER_BODY_W_STERNUM,
  "upper_body":           MvnSensorSetup.UPPER_BODY,
  "upper_body_no_hands":  MvnSensorSetup.UPPER_BODY_NO_HANDS,
}

#####################################################################
#####################################################################
# A structure to store MVN Analyze stream's medically certified data.
#####################################################################
#####################################################################
class MvnAnalyzeStream(Stream):
  def __init__(self,
               mvn_setup: str,
               sampling_rate_hz: int = 60,
               is_euler: bool = False,
               is_quaternion: bool = False,
               is_joint_angles: bool = False,
               is_linear_segments: bool = False,
               is_angular_segments: bool = False,
               is_motion_trackers: bool = False,
               is_com: bool = False,
               is_time_code: bool = False,
               timesteps_before_solidified: int = 0,
               update_interval_ms: int = 100,
               transmission_delay_period_s: int | None = None,
               **_) -> None:

    super().__init__()
    self._mvn_segment_setup = MVN_SEGMENT_SETUP[mvn_setup]
    self._mvn_joint_setup = MVN_JOINT_SETUP[mvn_setup]
    self._mvn_sensor_setup = MVN_SENSOR_SETUP[mvn_setup]

    self._num_segments = len(self._mvn_segment_setup)
    self._num_sensors = len(self._mvn_sensor_setup)
    self._num_joints = len(self._mvn_joint_setup)

    self._sampling_rate_hz = sampling_rate_hz
    self._is_euler = is_euler
    self._is_quaternion = is_quaternion
    self._is_joint_angles = is_joint_angles
    self._is_linear_segments = is_linear_segments
    self._is_angular_segments = is_angular_segments
    self._is_motion_trackers = is_motion_trackers
    self._is_com = is_com
    self._is_time_code = is_time_code

    self._transmission_delay_period_s = transmission_delay_period_s
    self._timesteps_before_solidified = timesteps_before_solidified
    self._update_interval_ms = update_interval_ms

    self._define_data_notes()

    # Segment positions and orientations.
    if is_euler or is_quaternion:
      self.add_stream(device_name='xsens-pose',
                      stream_name='position',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-pose']['position'])
      self.add_stream(device_name='xsens-pose',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-pose']['counter'])
      self.add_stream(device_name='xsens-pose',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-pose']['time_since_start_s'])
    if is_euler:
      self.add_stream(device_name='xsens-pose',
                      stream_name='euler',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-pose']['euler'])
    if is_quaternion:
      self.add_stream(device_name='xsens-pose',
                      stream_name='quaternion',
                      data_type='float32',
                      sample_size=(self._num_segments, 4),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-pose']['quaternion'])
    
    # Joint angles.
    if is_joint_angles:
      self.add_stream(device_name='xsens-joints',
                      stream_name='angle',
                      data_type='float32',
                      sample_size=(self._num_joints, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-joints']['angle'])
      self.add_stream(device_name='xsens-joints',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-joints']['counter'])
      self.add_stream(device_name='xsens-joints',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-joints']['time_since_start_s'])

    # Center of mass dynamics.
    if is_com:
      self.add_stream(device_name='xsens-com',
                      stream_name='position',
                      data_type='float32',
                      sample_size=(3,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-com']['position'])
      self.add_stream(device_name='xsens-com',
                      stream_name='velocity',
                      data_type='float32',
                      sample_size=(3,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-com']['velocity'])
      self.add_stream(device_name='xsens-com',
                      stream_name='acceleration',
                      data_type='float32',
                      sample_size=(3,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-com']['acceleration'])
      self.add_stream(device_name='xsens-com',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-com']['counter'])
      self.add_stream(device_name='xsens-com',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-com']['time_since_start_s'])

    # Liner segment kinematics.
    if is_linear_segments:
      self.add_stream(device_name='xsens-linear-segments',
                      stream_name='position',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-linear-segments']['position'])
      self.add_stream(device_name='xsens-linear-segments',
                      stream_name='velocity',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-linear-segments']['velocity'])
      self.add_stream(device_name='xsens-linear-segments',
                      stream_name='acceleration',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-linear-segments']['acceleration'])
      self.add_stream(device_name='xsens-linear-segments',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-linear-segments']['counter'])
      self.add_stream(device_name='xsens-linear-segments',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-linear-segments']['time_since_start_s'])
    
    # Angular segment cinematics.
    if is_angular_segments:
      self.add_stream(device_name='xsens-angular-segments',
                      stream_name='quaternion',
                      data_type='float32',
                      sample_size=(self._num_segments, 4),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-angular-segments']['quaternion'])
      self.add_stream(device_name='xsens-angular-segments',
                      stream_name='velocity',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-angular-segments']['velocity'])
      self.add_stream(device_name='xsens-angular-segments',
                      stream_name='acceleration',
                      data_type='float32',
                      sample_size=(self._num_segments, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-angular-segments']['acceleration'])
      self.add_stream(device_name='xsens-angular-segments',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-angular-segments']['counter'])
      self.add_stream(device_name='xsens-angular-segments',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-angular-segments']['time_since_start_s'])

    # Sensor kinematics.
    if is_motion_trackers:
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='quaternion',
                      data_type='float32',
                      sample_size=(self._num_sensors, 4),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['quaternion'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='free_acceleration',
                      data_type='float32',
                      sample_size=(self._num_sensors, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['free_acceleration'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='acceleration',
                      data_type='float32',
                      sample_size=(self._num_sensors, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['acceleration'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='gyroscope',
                      data_type='float32',
                      sample_size=(self._num_sensors, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['gyroscope'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='magnetometer',
                      data_type='float32',
                      sample_size=(self._num_sensors, 3),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['magnetometer'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='counter',
                      data_type='int32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['counter'])
      self.add_stream(device_name='xsens-motion-trackers',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-motion-trackers']['time_since_start_s'])

    # Time codes sent from the Xsens device.
    if is_time_code:
      self.add_stream(device_name='xsens-time',
                      stream_name='timestamp_s',
                      data_type='float64',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-time']['timestamp_s'])
      self.add_stream(device_name='xsens-time',
                      stream_name='counter',
                      data_type='uint32',
                      sample_size=(1,),
                      is_measure_rate_hz=True,
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-time']['counter'])
      self.add_stream(device_name='xsens-time',
                      stream_name='time_since_start_s',
                      data_type='float32',
                      sample_size=(1,),
                      sampling_rate_hz=self._sampling_rate_hz,
                      data_notes=self._data_notes['xsens-time']['time_since_start_s'])


  def get_fps(self) -> dict[str, float | None]:
    return {device_name: super()._get_fps(device_name, 'counter') for device_name in self._streams_info.keys()}
  

  def build_visulizer(self) -> dbc.Row:
    acceleration_plot = LinePlotVisualizer(stream=self,
                                           unique_id='xsens_acc',
                                           data_path={'xsens-motion-trackers': ['acceleration']},
                                           legend_names=list(self._mvn_sensor_setup.values()),
                                           plot_duration_timesteps=self._timesteps_before_solidified,
                                           update_interval_ms=self._update_interval_ms,
                                           col_width=6)
    return dbc.Row([acceleration_plot])


  def _define_data_notes(self):
    self._data_notes = {}
    self._data_notes.setdefault('xsens-pose', {})
    self._data_notes.setdefault('xsens-joints', {})
    self._data_notes.setdefault('xsens-com', {})
    self._data_notes.setdefault('xsens-linear-segments', {})
    self._data_notes.setdefault('xsens-angular-segments', {})
    self._data_notes.setdefault('xsens-motion-trackers', {})
    self._data_notes.setdefault('xsens-time', {})

    # 3D Pose.
    self._data_notes['xsens-pose']['position'] = OrderedDict([
      ('Description', 'Global position of segments in the Z-up right-handed coordinate system'),
      ('Units', 'cm'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-pose']['euler'] = OrderedDict([
      ('Description', 'Global rotation of segments in degrees in the Z-up right-handed coordinate system'),
      ('Units', 'degrees'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-pose']['quaternion'] = OrderedDict([
      ('Description', 'Global orientation of segments as unit quaternions in the Z-up right-handed coordinate system'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-pose']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-pose']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Joints.
    self._data_notes['xsens-joints']['angle'] = OrderedDict([
      ('Description', 'Joint angles between adjoint segments in the Z-Up, right-handed coordinate system'),
      ('Units', 'degrees'),
      (Stream.metadata_data_headings_key, {joint_details.name: joint_details.description for joint_details in self._mvn_joint_setup.values()}),
    ])
    self._data_notes['xsens-joints']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-joints']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Center of mass.
    self._data_notes['xsens-com']['position'] = OrderedDict([
      ('Description', '3D position of the Center of Mass in the Z-up, right-handed coordinate system'),
      ('Units', 'centimeter'),
    ])
    self._data_notes['xsens-com']['velocity'] = OrderedDict([
      ('Description', 'Velocity of the Center of Mass in the Z-up, right-handed coordinate system'),
      ('Units', 'centimeter/second'),
    ])
    self._data_notes['xsens-com']['acceleration'] = OrderedDict([
      ('Description', 'Linear acceleration of the Center of Mass in the Z-up, right-handed coordinate system'),
      ('Units', 'centimeter/second^2'),
    ])
    self._data_notes['xsens-com']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-com']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Linear segments.
    self._data_notes['xsens-linear-segments']['position'] = OrderedDict([
      ('Description', '3D coordinates of the segment in the global coordinate system'),
      ('Units', 'centimeter'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-linear-segments']['velocity'] = OrderedDict([
      ('Description', 'Linear velocity of the segment in the global coordinate system'),
      ('Units', 'centimeter/second'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-linear-segments']['acceleration'] = OrderedDict([
      ('Description', 'Linear acceleration of the segment in the global coordinate system'),
      ('Units', 'centimeter/second^2'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-linear-segments']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-linear-segments']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Angular segments.
    self._data_notes['xsens-angular-segments']['quaternion'] = OrderedDict([
      ('Description', 'Quaternion orientation vector of the segment with respect to the global coordinate system'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-angular-segments']['velocity'] = OrderedDict([
      ('Description', 'Angular velocity of the segment'),
      ('Units', 'degree/second'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-angular-segments']['acceleration'] = OrderedDict([
      ('Description', 'Angular acceleration of the segment'),
      ('Units', 'degree/second^2'),
      (Stream.metadata_data_headings_key, list(self._mvn_segment_setup.values())),
    ])
    self._data_notes['xsens-angular-segments']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-angular-segments']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Sensors.
    self._data_notes['xsens-motion-trackers']['quaternion'] = OrderedDict([
      ('Description', 'Quaternion orientation vector of the sensor with respect to the global coordinate system'),
      (Stream.metadata_data_headings_key, list(self._mvn_sensor_setup.values())),
    ])
    self._data_notes['xsens-motion-trackers']['free_acceleration'] = OrderedDict([
      ('Description', 'Local linear acceleration of the IMU, with the gravitational component subtracted'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._mvn_sensor_setup.values())),
    ])
    self._data_notes['xsens-motion-trackers']['acceleration'] = OrderedDict([
      ('Description', 'Local raw linear acceleration of the IMU'),
      ('Units', 'meter/second^2'),
      (Stream.metadata_data_headings_key, list(self._mvn_sensor_setup.values())),
    ])
    self._data_notes['xsens-motion-trackers']['gyroscope'] = OrderedDict([
      ('Description', 'Local raw angular velocity of the IMU'),
      ('Units', 'meter/second'),
      (Stream.metadata_data_headings_key, list(self._mvn_sensor_setup.values())),
    ])
    self._data_notes['xsens-motion-trackers']['magnetometer'] = OrderedDict([
      ('Description', 'Local raw magnetic field of the IMU, normalized against the magnetic field at the factory'),
      ('Units', 'a.u. w.r.t. magnetic field at the calibration site'),
      (Stream.metadata_data_headings_key, list(self._mvn_sensor_setup.values())),
    ])
    self._data_notes['xsens-motion-trackers']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-motion-trackers']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])

    # Time.
    self._data_notes['xsens-time']['counter'] = OrderedDict([
      ('Description', 'Index of the sample provisioned by MVN Analyze'),
    ])
    self._data_notes['xsens-time']['timestamp_s'] = OrderedDict([
      ('Description', 'Time of sampling of the data by MVN Analyze'),
    ])
    self._data_notes['xsens-time']['time_since_start_s'] = OrderedDict([
      ('Description', 'MVN timecode from the datagram metadata'),
    ])
