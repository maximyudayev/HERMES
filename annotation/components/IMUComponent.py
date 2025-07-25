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

from .BaseComponent import BaseComponent
from utils.gui_utils import app
from dash import Output, Input, State, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import h5py


class IMUComponent(BaseComponent):
  def __init__(self,
               hdf5_path: str,
               data_path: str,
               timestamp_path: str,
               unique_id: str,
               legend_name: str,
               sensor_type: str,  # 'accelerometer', 'gyroscope', or 'magnetometer'
               plot_window_seconds: float = 1.0,
               sampling_rate: float = 60.0,
               project="AidWear",
               col_width: int = 3):
    super().__init__(unique_id=unique_id, col_width=col_width)

    self._hdf5_path = hdf5_path
    self._data_path = data_path
    self._timestamp_path = timestamp_path
    self._legend_name = legend_name
    self._sensor_type = sensor_type
    self._plot_window_seconds = plot_window_seconds
    self._sampling_rate = sampling_rate

    # Joint names corresponding to the 17 sensors for AidWear and 7 for RevalExo (taken from the description from the HDF5 file)
    if project == "AidWear":
        self._joint_names = ['Pelvis', 'T8', 'Head', 'Right Shoulder', 'Right Upper Arm', 
                            'Right Forearm', 'Right Hand', 'Left Shoulder', 'Left Upper Arm', 
                            'Left Forearm', 'Left Hand', 'Right Upper Leg', 'Right Lower Leg', 
                            'Right Foot', 'Left Upper Leg', 'Left Lower Leg', 'Left Foot']
        self._expected_joints = 17
    elif project == "RevalExo":
        self._joint_names = ['Pelvis', 'Right Upper Leg', 'Right Lower Leg', 
                            'Right Foot', 'Left Upper Leg', 'Left Lower Leg', 'Left Foot']
        self._expected_joints = 7
    else:
        raise ValueError(f"Unknown project: {project}")

    # Set units based on sensor type
    if sensor_type == 'accelerometer':
      self._y_units = 'm/s²'
    elif sensor_type == 'gyroscope':
      self._y_units = 'm/s'
    elif sensor_type == 'magnetometer':
      self._y_units = 'μT'
    else:
      self._y_units = 'units'

    # Read HDF5 data
    self._read_data()

    # Calculate symmetric y-axis scaling using percentiles
    # Use percentiles to handle outliers, then create symmetric scale
    # This could truncate extreme values, but double clicking on the graph will bring them back to view
    percentile_low = np.percentile(self._data, 1)
    percentile_high = np.percentile(self._data, 99)

    # Find the maximum absolute value with 2x factor for more headroom
    max_abs_value = max(abs(2*percentile_low), abs(2*percentile_high))

    # Create symmetric range around zero 
    self._y_min = -max_abs_value
    self._y_max = max_abs_value
    self._y_range = [-max_abs_value, max_abs_value]

    # Initialize truncation points
    self._start_idx = 0
    self._end_idx = len(self._timestamps) - 1

    # Current selected joint (default to first - pelvis)
    self._selected_joint_idx = 0

    # Create layout
    self._dropdown = dcc.Dropdown(
      id=f"{self._unique_id}-joint-dropdown",
      options=[{"label": joint, "value": i} for i, joint in enumerate(self._joint_names)],
      value=0,
      clearable=False,
      style={'marginBottom': '10px'}
    )

    self._graph = dcc.Graph(
      id=f"{self._unique_id}-imu-plot",
      config={'displayModeBar': False},
      clear_on_unhover=True
    )

    self._timestamp_display = html.Div(
      id=f"{self._unique_id}-timestamp",
      className="text-center small text-muted",
      style={'fontSize': '12px'}
    )

    self._layout = dbc.Col([
      html.H6(self._legend_name, className="text-center mb-2"),
      self._dropdown,
      self._graph,
      self._timestamp_display
    ], width=self._col_width)

    self._activate_callbacks()


  def _read_data(self):
    """Read IMU data and timestamps from HDF5"""
    with h5py.File(self._hdf5_path, 'r') as hdf5:
      # Read timestamps
      if self._timestamp_path in hdf5:
        self._timestamps = hdf5[self._timestamp_path][:]
        self._first_timestamp = float(self._timestamps[0])
        self._last_timestamp = float(self._timestamps[-1])
      else:
        raise ValueError(f"Timestamp path {self._timestamp_path} not found in HDF5")

      # Read data
      if self._data_path in hdf5:
        self._data = hdf5[self._data_path][:]
        # Expected shape: (num_timestamps, 17 joints, 3 axes) - TODO: NUM_JOINTS FLEXIBLE FOR REVALEXO
        if len(self._data.shape) != 3 or self._data.shape[1] != self._expected_joints or self._data.shape[2] != 3:
          raise ValueError(f"Expected IMU data shape (timestamps, {self._expected_joints}, 3), got {self._data.shape}")

        # Verify data and timestamp lengths match
        if len(self._data) != len(self._timestamps):
          print(f"Warning: Data length ({len(self._data)}) != timestamp length ({len(self._timestamps)})")
          min_length = min(len(self._data), len(self._timestamps))
          self._data = self._data[:min_length]
          self._timestamps = self._timestamps[:min_length]
      else:
        raise ValueError(f"Data path {self._data_path} not found in HDF5")


  def get_sync_info(self):
    """Return synchronization info for this component"""
    return {
      'type': 'imu',
      'unique_id': self._unique_id,
      'first_timestamp': self._first_timestamp,
      'last_timestamp': self._last_timestamp,
      'timestamps': self._timestamps
    }


  # ONLY USED FOR INITIAL DATA DISPLAY 
  def set_truncation_points(self, start_idx: int, end_idx: int):
    """Set truncation points"""
    self._start_idx = int(max(0, start_idx))
    self._end_idx = int(min(len(self._timestamps) - 1, end_idx))
    print(f"{self._legend_name}: Start index = {self._start_idx}")


  # Used to sync with the frame slider update, OFFSET ALREADY APPLIED - SHARED WITH SKELETON
  def get_timestamp_for_sync(self, sync_timestamp: float) -> int:
    """Find the index closest to a given timestamp with offset"""
    if self._timestamps is not None:
      time_diffs = np.abs(self._timestamps - sync_timestamp)
      closest_idx = np.argmin(time_diffs)
      # Apply offset
      offset_idx = closest_idx + self._sync_offset
      # Ensure within bounds
      offset_idx = max(0, min(len(self._timestamps) - 1, offset_idx))
      return int(offset_idx)
    return 0


  def _create_figure(self, center_idx: int, joint_idx: int):
    """Create the line plot figure for the given center index and joint"""
    # Ensure center_idx is within bounds
    center_idx = max(0, min(center_idx, len(self._data) - 1))

    # Calculate window bounds
    window_samples = int(self._plot_window_seconds * self._sampling_rate)
    half_window = window_samples // 2

    start_idx = max(0, center_idx - half_window)
    end_idx = min(len(self._data) - 1, center_idx + half_window)

    # Get data slice for the selected joint
    data_slice = self._data[start_idx:end_idx+1, joint_idx, :]  # Shape: (samples, 3)
    time_slice = np.arange(len(data_slice)) / self._sampling_rate

    # Calculate where the red line should be (current position in window)
    red_line_position = (center_idx - start_idx) / self._sampling_rate

    # Create subplots for X, Y, Z
    fig = make_subplots(
      rows=3,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.02,
      subplot_titles=['X', 'Y', 'Z']
    )

    # Add traces for each axis
    colors = ['blue', 'green', 'red']
    axes = ['X', 'Y', 'Z']

    for i in range(3):
      fig.add_trace(
        go.Scatter(
          x=time_slice,
          y=data_slice[:, i],
          mode='lines',
          name=axes[i],
          line=dict(width=1, color=colors[i])
        ),
        row=i+1,
        col=1
      )

      # Add vertical line at current position
      fig.add_vline(
        x=red_line_position,
        line_dash="dash",
        line_color="red",
        row=i+1,
        col=1
      )

    # Update layout
    title = f"{self._legend_name} - {self._joint_names[joint_idx]}"
    if self._sync_offset != 0:
      title += f" [offset: {self._sync_offset:+d}]"

    fig.update_layout(
      title_text=title,
      showlegend=False,
      margin=dict(l=50, r=20, t=60, b=20),
      height=400,
      width=400
    )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    # Update y-axes with consistent range
    for i in range(3):
      fig.update_yaxes(
        title_text=self._y_units,
        range=self._y_range,
        row=i+1,
        col=1
      )

    return fig


  def _activate_callbacks(self):
    @app.callback(
      Output(f"{self._unique_id}-imu-plot", "figure"),
      Output(f"{self._unique_id}-timestamp", "children"),
      Input("sync-timestamp", "data"),
      Input(f"{self._unique_id}-joint-dropdown", "value"),
      Input("offset-update-trigger", "data"),
      prevent_initial_call=False
    )
    def update_plot(sync_timestamp, selected_joint, offset_trigger):
      try:
        # Update selected joint
        joint_idx = selected_joint if selected_joint is not None else 0

        if sync_timestamp is not None:
          # Find the index matching the sync timestamp
          current_idx = self.get_timestamp_for_sync(sync_timestamp)

          # Create the figure
          fig = self._create_figure(current_idx, joint_idx)

          # Get timestamp for display
          timestamp = self._timestamps[current_idx] if current_idx < len(self._timestamps) else 0
          timestamp_float = float(timestamp)
          timestamp_text = f"timestamp_s: {timestamp_float:.7f} (index: {current_idx})"

          return fig, timestamp_text
        else:
          # Show initial data at start_idx if no sync
          fig = self._create_figure(self._start_idx, joint_idx)
          timestamp = self._timestamps[self._start_idx] if self._start_idx < len(self._timestamps) else 0
          timestamp_float = float(timestamp)
          timestamp_text = f"timestamp_s: {timestamp_float:.7f} (index: {self._start_idx})"
          return fig, timestamp_text

      except Exception as e:
        print(f"Error updating IMU plot: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure(), "Error"
