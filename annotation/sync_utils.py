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

from components import * 
import numpy as np
from typing import List, Mapping, Tuple

# Function to check the overlapping window for all the components, used for RevalExo for example to trim out the initial frames from video that starts way earlier than other components 
def find_overlap_window(all_sync_infos: List[dict]) -> Tuple[float, float]:
  """Find the time window where all components have data."""
  start_times = []
  end_times = []
  
  for info in all_sync_infos:
    if 'timestamps' in info and len(info['timestamps']) > 0:
      start_times.append(float(info['timestamps'][0]))
      end_times.append(float(info['timestamps'][-1]))
    elif 'toa_s' in info and len(info['toa_s']) > 0:
      start_times.append(float(info['toa_s'][0]))
      end_times.append(float(info['toa_s'][-1]))
  
  if not start_times or not end_times:
    raise ValueError("No valid timestamp data found in components")
  
  overlap_start = max(start_times)
  overlap_end = min(end_times)
  
  if overlap_start >= overlap_end:
    raise ValueError("No time overlap between all components")
  
  return overlap_start, overlap_end

def calculate_truncation_points(
  camera_components: List[VideoComponent],
  eye_component: GazeComponent | None = None,
  emg_components: List[LinePlotComponent] | None = None,
  skeleton_components: List[SkeletonComponent] | None = None,
  insole_components: List[LinePlotComponent] | None = None,
  imu_components: List[IMUComponent] | None = None,
  baseline_frame: int = 100) -> Mapping:
  """
  Calculate truncation points using reference camera (Camera 1) as baseline.
  
  Args:
    camera_components: List of camera VideoComponent instances
    eye_component: Eye camera VideoComponent instance (can be None)
    emg_components: List of EMG LinePlotComponent instances (optional)
    skeleton_components: List of SkeletonComponent instances (optional)
    insole_components: List of insole LinePlotComponent instances (optional)
    imu_components: List of IMUComponent instances (optional)
    baseline_frame: Reference camera frame to use as synchronization reference (default: 100)
  
  Returns:
    dict: Truncation points for each component {'component_id': (start_frame/idx, end_frame/idx)}
  """
  truncation_points = {}

  if not camera_components:
    raise ValueError("Camera components are required for synchronization")

  # Find the reference camera (Camera 1)
  reference_camera = None
  for cam in camera_components:
    if cam._is_reference_camera:
      reference_camera = cam
      break

  if not reference_camera:
    raise ValueError("No reference camera found. Please mark one camera as reference.")

  # Get sync info from all components
  camera_infos = [cam.get_sync_info() for cam in camera_components]
  eye_info = eye_component.get_sync_info() if eye_component else None

  # Get EMG sync info if available
  emg_infos = []
  if emg_components:
    emg_infos = [emg.get_sync_info() for emg in emg_components]

  # Get skeleton sync info if available
  skeleton_infos = []
  if skeleton_components:
    skeleton_infos = [skeleton.get_sync_info() for skeleton in skeleton_components]

  # Get insole sync info if available
  insole_infos = []
  insole_components = insole_components or []
  if insole_components:
    insole_infos = [insole.get_sync_info() for insole in insole_components]

  # Get IMU sync info if available
  imu_infos = []
  imu_components = imu_components or []
  if imu_components:
    imu_infos = [imu.get_sync_info() for imu in imu_components]

  all_sync_infos = []
  all_sync_infos.extend(camera_infos)
  if eye_info:
    all_sync_infos.append(eye_info)
  all_sync_infos.extend(emg_infos)
  all_sync_infos.extend(skeleton_infos)
  all_sync_infos.extend(insole_infos)
  all_sync_infos.extend(imu_infos)

  # Find overlap window
  overlap_start, overlap_end = find_overlap_window(all_sync_infos)
  overlap_duration = overlap_end - overlap_start
  print(f"Data overlap window: {overlap_start} - {overlap_end} (duration: {overlap_duration}s)")

  # Get the timestamp at the baseline frame for reference camera
  ref_cam_info = next(info for info in camera_infos if info['unique_id'] == reference_camera._unique_id)
  ref_timestamps = ref_cam_info['toa_s'] # Using toa_s from reference camera as the baseline sync timestamp to synchronize all other components

  # CHANGED THE FOLLOWING TO USE THE OVERLAP WINDOW INSTEAD OF THE FULL VIDEO LENGTH FROM BEFORE
  # Find the first frame in reference camera that's within the overlap window
  ref_start_idx = 0
  for i, ts in enumerate(ref_timestamps):
    if float(ts) >= overlap_start:
      ref_start_idx = i
      break
  
  # Find the last frame in reference camera that's within the overlap window
  ref_end_idx = len(ref_timestamps) - 1
  for i in range(len(ref_timestamps) - 1, -1, -1):
    if float(ref_timestamps[i]) <= overlap_end:
      ref_end_idx = i
      break
  
  # Add baseline_frame offset from the start of overlap
  sync_frame = ref_start_idx + baseline_frame
  if sync_frame > ref_end_idx:
    sync_frame = ref_end_idx

  baseline_timestamp = ref_timestamps[sync_frame]
  print(f"\nSynchronization using reference camera ({reference_camera._unique_id}) frame {baseline_frame} as baseline")
  print(f"Reference camera baseline timestamp: {baseline_timestamp}")


  # Find closest frame in each camera relative to the baseline timestamp
  # ALSO TRIED FRAME_SEQUENCE_ID, BUT THE DIFFERENCE SEEMED MINIMAL (1-2 FRAMES), IT IS LIKELY THAT EITHER OF THE TWO CAN BE USED
  # IT WAS NOT POSSIBLE TO DETERMINE WHICH ONE IS BETTER, SINCE FINDING THE EXACT MATCHING FRAME ACROSS CAMERAS IS DIFFICULT
  camera_sync_frames = {}
  for cam_idx, (cam, cam_info) in enumerate(zip(camera_components, camera_infos)):
    toa_s = cam_info['toa_s']

    # Find closest toa_s to baseline timestamp
    time_diffs = np.abs(toa_s - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = toa_s[closest_idx]

    camera_sync_frames[cam._unique_id] = {
        'sync_frame': closest_idx,
        'sync_timestamp': closest_timestamp,
        'time_diff': time_diffs[closest_idx]
    }
    print(f"Camera {cam._unique_id}: frame {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")

  # Find closest frame in eye camera if available
  if eye_component and eye_info:
    eye_timestamps = eye_info['timestamps']
    time_diffs = np.abs(eye_timestamps - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = eye_timestamps[closest_idx]

    print(f"Eye camera: frame {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")
    truncation_points[str(eye_component._unique_id)] = (closest_idx, len(eye_timestamps) - 1)

  # Find closest index in each EMG component
  emg_sync_indices = {}
  for emg_idx, (emg, emg_info) in enumerate(zip(emg_components or [], emg_infos)):
    timestamps = emg_info['timestamps']

    # For EMG with burst data, timestamps are the burst arrival times
    time_diffs = np.abs(timestamps - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = timestamps[closest_idx]

    emg_sync_indices[emg._unique_id] = {
      'sync_idx': closest_idx,
      'sync_timestamp': closest_timestamp,
      'time_diff': time_diffs[closest_idx]
    }
    print(f"EMG {emg._unique_id}: burst index {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")

  # Find closest index in each skeleton component
  skeleton_sync_indices = {}
  for skeleton_idx, (skeleton, skeleton_info) in enumerate(zip(skeleton_components or [], skeleton_infos)):
    timestamps = skeleton_info['timestamps']

    # Find closest timestamp to baseline
    time_diffs = np.abs(timestamps - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = timestamps[closest_idx]

    skeleton_sync_indices[skeleton._unique_id] = {
      'sync_idx': closest_idx,
      'sync_timestamp': closest_timestamp,
      'time_diff': time_diffs[closest_idx]
    }
    print(f"Skeleton {skeleton._unique_id}: index {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")

  # Find closest index in each insole component
  insole_sync_indices = {}
  for insole_idx, (insole, insole_info) in enumerate(zip(insole_components, insole_infos)):
    timestamps = insole_info['timestamps']

    # Find closest timestamp to baseline
    time_diffs = np.abs(timestamps - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = timestamps[closest_idx]

    insole_sync_indices[insole._unique_id] = {
      'sync_idx': closest_idx,
      'sync_timestamp': closest_timestamp,
      'time_diff': time_diffs[closest_idx]
    }
    print(f"Insole {insole._unique_id}: index {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")

  # Find closest index in each IMU component
  imu_sync_indices = {}
  for imu_idx, (imu, imu_info) in enumerate(zip(imu_components, imu_infos)):
    timestamps = imu_info['timestamps']

    # Find closest timestamp to baseline
    time_diffs = np.abs(timestamps - baseline_timestamp)
    closest_idx = np.argmin(time_diffs)
    closest_timestamp = timestamps[closest_idx]

    imu_sync_indices[imu._unique_id] = {
      'sync_idx': closest_idx,
      'sync_timestamp': closest_timestamp,
      'time_diff': time_diffs[closest_idx]
    }
    print(f"IMU {imu._unique_id}: index {closest_idx}, timestamp {closest_timestamp}, diff {time_diffs[closest_idx]}s")

  # Set truncation points for cameras
  for cam in camera_components:
      sync_info = camera_sync_frames[cam._unique_id]
      if cam._unique_id == reference_camera._unique_id:
          # Reference camera starts at overlap window start
          start_frame = ref_start_idx
          end_frame = ref_end_idx
      else:
          # Other cameras use their sync frame
          start_frame = sync_info['sync_frame']
          cam_info = next(info for info in camera_infos if info['unique_id'] == cam._unique_id)
          end_frame = len(cam_info['toa_s']) - 1
      
      truncation_points[str(cam._unique_id)] = (start_frame, end_frame)

  # Set truncation points for EMG components
  for emg in (emg_components or []):
    sync_info = emg_sync_indices[emg._unique_id]
    start_idx = sync_info['sync_idx']
    # End index is the last burst
    emg_info = next(info for info in emg_infos if info['unique_id'] == emg._unique_id)
    end_idx = len(emg_info['timestamps']) - 1
    truncation_points[str(emg._unique_id)] = (start_idx, end_idx)
  
  # Set truncation points for skeleton components
  for skeleton in (skeleton_components or []):
    sync_info = skeleton_sync_indices[skeleton._unique_id]
    start_idx = sync_info['sync_idx']
    # End index is the last sample
    skeleton_info = next(info for info in skeleton_infos if info['unique_id'] == skeleton._unique_id)
    end_idx = len(skeleton_info['timestamps']) - 1
    truncation_points[str(skeleton._unique_id)] = (start_idx, end_idx)

  # Set truncation points for insole components
  for insole in insole_components:
    sync_info = insole_sync_indices[insole._unique_id]
    start_idx = sync_info['sync_idx']
    # End index is the last sample
    insole_info = next(info for info in insole_infos if info['unique_id'] == insole._unique_id)
    end_idx = len(insole_info['timestamps']) - 1
    truncation_points[str(insole._unique_id)] = (start_idx, end_idx)

  # Set truncation points for IMU components
  for imu in imu_components:
    sync_info = imu_sync_indices[imu._unique_id]
    start_idx = sync_info['sync_idx']
    # End index is the last sample
    imu_info = next(info for info in imu_infos if info['unique_id'] == imu._unique_id)
    end_idx = len(imu_info['timestamps']) - 1
    truncation_points[str(imu._unique_id)] = (start_idx, end_idx)

  # For simplicity, not adjusting end points to ensure equal length
  # as different data types may have different sampling rates

  return truncation_points


def apply_truncation(components, truncation_points):
  """
  Apply truncation points to all components.
  
  Args:
      components: List of VideoComponent, LinePlotComponent, SkeletonComponent, and IMUComponent instances
      truncation_points: Dict of truncation points from calculate_truncation_points
  """
  for component in components:
    # Use string key to look up truncation points
    key = str(component._unique_id)
    if key in truncation_points:
      start, end = truncation_points[key]
      component.set_truncation_points(start, end)
