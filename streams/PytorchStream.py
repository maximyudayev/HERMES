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


##################################################
##################################################
# A structure to store PyTorch prediction outputs.
# TODO: use user parameters to specify model 
#   output configuration 
#   (i.e. classifier, regressor, embedding, etc.)
##################################################
##################################################
class PytorchStream(Stream):
  def __init__(self,
               classes: list[str],
               sampling_rate_hz: float,
               **_) -> None:
    super().__init__()

    self._classes = classes
    self._define_data_notes()

    self.add_stream(device_name='pytorch-worker',
                    stream_name='prediction',
                    data_type='uint16',
                    sample_size=(1,),
                    sampling_rate_hz=sampling_rate_hz,
                    is_measure_rate_hz=True,
                    data_notes=self._data_notes['pytorch-worker']['prediction'])
    self.add_stream(device_name='pytorch-worker',
                    stream_name='logits',
                    data_type='float64',
                    sample_size=(len(classes),),
                    data_notes=self._data_notes['pytorch-worker']['logits'])
    self.add_stream(device_name='pytorch-worker',
                    stream_name='inference_latency_s',
                    data_type='float64',
                    sample_size=(1,),
                    data_notes=self._data_notes['pytorch-worker']['inference_latency_s'])

  
  def get_fps(self) -> dict[str, float | None]:
    return {'pytorch-worker': super()._get_fps('pytorch-worker', 'prediction')}


  def build_visulizer(self) -> dbc.Row | None:
    return super().build_visulizer()
  

  def _define_data_notes(self) -> None:
    self._data_notes = {}
    self._data_notes.setdefault('pytorch-worker', {})

    self._data_notes['pytorch-worker']['logits'] = OrderedDict([
      ('Description', 'Probability vector'),
      ('Range', '[0,1]'),
      (Stream.metadata_data_headings_key, self._classes),
    ])
    self._data_notes['pytorch-worker']['prediction'] = OrderedDict([
      ('Description', 'Label of the most likely class prediction'),
    ])
    self._data_notes['pytorch-worker']['inference_latency_s'] = OrderedDict([
      ('Description', 'Amount of time it took for the forward pass for the new sample w.r.t. system clock'),
      ('Units', 'seconds'),
    ])
