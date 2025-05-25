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

from visualizers import ExperimentControlVisualizer


################################################
################################################
# A structure to store Experiment stream's data.
################################################
################################################
class ExperimentControlStream(Stream):
  def __init__(self, 
               activities: list[str],
               sampling_rate_hz: int = 0,
               **_) -> None:
    super().__init__()
    self._activities = activities
    self._define_data_notes()

    self.add_stream(device_name='experiment',
                    stream_name='activity',
                    data_type='S26',
                    sample_size=[1],
                    sampling_rate_hz=sampling_rate_hz)


  def build_visulizer(self) -> dbc.Row:
    control_gui = ExperimentControlVisualizer(stream=self,
                                              activities=self._activities,
                                              col_width=6)
    return dbc.Row([control_gui.layout])


  def get_fps(self) -> dict[str, float | None]:
    return {'experiment': None}


  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('experiment', {})

    self._data_notes['experiment']['activity'] = OrderedDict([
      ('Description', 'Label of the performed activity, marked during the trial by the researcher. '
                      '[0,%d], corresponding to %s'.format(len(self._activities)-1, self._activities))
    ])
