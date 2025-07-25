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
from dash import Output, Input, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import h5py


class SkeletonComponent(BaseComponent):
  def __init__(self,
               hdf5_path: str,
               position_path: str,
               timestamp_path: str,
               unique_id: str,
               legend_name: str,
               project="AidWear",
               col_width: int = 3):
    super().__init__(unique_id=unique_id, col_width=col_width)

    self._hdf5_path = hdf5_path
    self._position_path = position_path
    self._timestamp_path = timestamp_path
    self._legend_name = legend_name

    # Segment names and indices (BASED ON THE HDF5 FILE DESCRIPTION)
    if project == "AidWear":
        self._segment_names = [
            'Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 
            'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
            'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',
            'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
            'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe']

        self._connections = [
            # Spine
            ('Pelvis', 'L5'), ('L5', 'L3'), ('L3', 'T12'), ('T12', 'T8'), ('T8', 'Neck'), ('Neck', 'Head'),
            # Right arm
            ('Neck', 'Right Shoulder'), ('Right Shoulder', 'Right Upper Arm'), 
            ('Right Upper Arm', 'Right Forearm'), ('Right Forearm', 'Right Hand'),
            # Left arm
            ('Neck', 'Left Shoulder'), ('Left Shoulder', 'Left Upper Arm'),
            ('Left Upper Arm', 'Left Forearm'), ('Left Forearm', 'Left Hand'),
            # Right leg
            ('Pelvis', 'Right Upper Leg'), ('Right Upper Leg', 'Right Lower Leg'),
            ('Right Lower Leg', 'Right Foot'), ('Right Foot', 'Right Toe'),
            # Left leg
            ('Pelvis', 'Left Upper Leg'), ('Left Upper Leg', 'Left Lower Leg'),
            ('Left Lower Leg', 'Left Foot'), ('Left Foot', 'Left Toe')
        ]
    elif project == "RevalExo":
        self._segment_names = [
            'Pelvis', 'L5', 'L3', 'Right Upper Leg', 'Right Lower Leg', 
            'Right Foot', 'Right Toe', 'Left Upper Leg', 'Left Lower Leg', 
            'Left Foot', 'Left Toe']

        self._connections = [
            # Spine (limited)
            ('Pelvis', 'L5'), ('L5', 'L3'),
            # Right leg
            ('Pelvis', 'Right Upper Leg'), ('Right Upper Leg', 'Right Lower Leg'),
            ('Right Lower Leg', 'Right Foot'), ('Right Foot', 'Right Toe'),
            # Left leg
            ('Pelvis', 'Left Upper Leg'), ('Left Upper Leg', 'Left Lower Leg'),
            ('Left Lower Leg', 'Left Foot'), ('Left Foot', 'Left Toe')
        ]
    else:
        raise ValueError(f"Unknown project: {project}")

    # Read data
    self._read_data()

    # Initialize truncation points
    self._start_idx = 0
    self._end_idx = len(self._timestamps) - 1

    # Create layout
    self._graph = dcc.Graph(
      id=f"{self._unique_id}-skeleton",
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
      self._graph,
      self._timestamp_display
    ], width=self._col_width)

    self._activate_callbacks()


  def _read_data(self):
    """Read position data and timestamps from HDF5"""
    with h5py.File(self._hdf5_path, 'r') as hdf5:
      # Read timestamps
      if self._timestamp_path in hdf5:
        self._timestamps = hdf5[self._timestamp_path][:]
        self._first_timestamp = float(self._timestamps[0])
        self._last_timestamp = float(self._timestamps[-1])
      else:
        raise ValueError(f"Timestamp path {self._timestamp_path} not found in HDF5")

      # Read position data
      if self._position_path in hdf5:
        self._positions = hdf5[self._position_path][:]
        # Expected shape: (num_frames, num_segments, 3)
        if len(self._positions.shape) != 3 or self._positions.shape[2] != 3:
          raise ValueError(f"Expected position data shape (frames, segments, 3), got {self._positions.shape}")

        # Verify data and timestamp lengths match
        if len(self._positions) != len(self._timestamps):
          print(f"Warning: Position data length ({len(self._positions)}) != timestamp length ({len(self._timestamps)})")
          min_length = min(len(self._positions), len(self._timestamps))
          self._positions = self._positions[:min_length]
          self._timestamps = self._timestamps[:min_length]
      else:
        raise ValueError(f"Position path {self._position_path} not found in HDF5")


  def get_sync_info(self):
    """Return synchronization info for this component"""
    return {
      'type': 'skeleton',
      'unique_id': self._unique_id,
      'first_timestamp': self._first_timestamp,
      'last_timestamp': self._last_timestamp,
      'timestamps': self._timestamps
    }


  def set_truncation_points(self, start_idx: int, end_idx: int):
    """Set truncation points for this data"""
    self._start_idx = int(max(0, start_idx))
    self._end_idx = int(min(len(self._timestamps) - 1, end_idx))
    print(f"{self._legend_name}: Start index = {self._start_idx}")


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


  def _create_figure(self, frame_idx: int):
    """Create the 3D skeleton figure for the given frame"""
    # Ensure frame_idx is within bounds
    frame_idx = max(0, min(frame_idx, len(self._positions) - 1))

    # Get positions for this frame
    positions = self._positions[frame_idx]  # Shape: (num_segments, 3)

    # Create 3D scatter plot
    fig = go.Figure()

    # Add joints
    fig.add_trace(go.Scatter3d(
      x=positions[:, 0],
      y=positions[:, 1],
      z=positions[:, 2],
      mode='markers',
      marker=dict(size=6, color='blue'),
      text=self._segment_names,
      hovertemplate='%{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>',
      name='Joints'
    ))

    # Add bones
    for conn in self._connections:
      start_idx = self._segment_names.index(conn[0])
      end_idx = self._segment_names.index(conn[1])

      fig.add_trace(go.Scatter3d(
          x=[positions[start_idx, 0], positions[end_idx, 0]],
          y=[positions[start_idx, 1], positions[end_idx, 1]],
          z=[positions[start_idx, 2], positions[end_idx, 2]],
          mode='lines',
          line=dict(color='gray', width=4),
          showlegend=False,
          hoverinfo='skip'
      ))

    # Update layout
    title = self._legend_name
    if self._sync_offset != 0:
      title += f" [offset: {self._sync_offset:+d}]"

    fig.update_layout(
      title_text=title,
      scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data',
        camera=dict(
          eye=dict(x=1.5, y=1.5, z=1.0)
        )
      ),
      showlegend=False,
      margin=dict(l=0, r=0, t=30, b=0),
      height=400,
      width=400
    )

    return fig


  def _activate_callbacks(self):
    @app.callback(
      Output(f"{self._unique_id}-skeleton", "figure"),
      Output(f"{self._unique_id}-timestamp", "children"),
      Input("sync-timestamp", "data"),
      Input("offset-update-trigger", "data"),
      prevent_initial_call=False
    )
    def update_skeleton(sync_timestamp, offset_trigger):
      try:
        if sync_timestamp is not None:
          # Find the index matching the sync timestamp
          current_idx = self.get_timestamp_for_sync(sync_timestamp)

          # Create the figure
          fig = self._create_figure(current_idx)

          # Get timestamp for display
          timestamp = self._timestamps[current_idx] if current_idx < len(self._timestamps) else 0
          timestamp_float = float(timestamp)
          timestamp_text = f"timestamp_s: {timestamp_float:.7f} (index: {current_idx})"

          return fig, timestamp_text
        else:
          # Show initial data at start_idx if no sync
          fig = self._create_figure(self._start_idx)
          timestamp = self._timestamps[self._start_idx] if self._start_idx < len(self._timestamps) else 0
          timestamp_float = float(timestamp)
          timestamp_text = f"timestamp_s: {timestamp_float:.7f} (index: {self._start_idx})"
          return fig, timestamp_text

      except Exception as e:
        print(f"Error updating skeleton: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure(), "Error"
