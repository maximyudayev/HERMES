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

from handlers.Tmsi.device.tmsi_device import TMSiDevice
from handlers.Tmsi.sample_data_server.sample_data import SampleData
from nodes.producers.Producer import Producer
from streams import TmsiStream

from handlers.Tmsi.tmsi_sdk import TMSiSDK
from handlers.Tmsi.device.tmsi_device_enums import DeviceInterfaceType, DeviceType, MeasurementType
from handlers.Tmsi.sample_data_server.sample_data_server import SampleDataServer
from handlers.Tmsi.tmsi_utilities.support_functions import array_to_matrix
from handlers.Tmsi.device.devices.saga.saga_API_enums import SagaBaseSampleRate
from handlers.Tmsi.device.tmsi_channel import ChannelType

import queue
from utils.print_utils import *
from utils.zmq_utils import *
from utils.time_utils import get_time
import time


########################################
########################################
# A class to interface TMSi SAGA device.
########################################
########################################
class TmsiStreamer(Producer):
  @classmethod
  def _log_source_tag(cls) -> str:
    return 'tmsi'


  def __init__(self,
               host_ip: str,
               logging_spec: dict,
               sampling_rate_hz: int = 20,
               port_pub: str = PORT_BACKEND,
               port_sync: str = PORT_SYNC_HOST,
               port_killsig: str = PORT_KILL,
               transmit_delay_sample_period_s: float = float('nan'),
               **_)-> None:
    
    stream_info = {
      "sampling_rate_hz": sampling_rate_hz
    }

    super().__init__(host_ip=host_ip,
                     stream_info=stream_info,
                     logging_spec=logging_spec,
                     sampling_rate_hz=sampling_rate_hz,
                     port_pub=port_pub,
                     port_sync=port_sync,
                     port_killsig=port_killsig,
                     transmit_delay_sample_period_s=transmit_delay_sample_period_s)


  @classmethod
  def create_stream(cls, stream_info: dict) -> TmsiStream:
    return TmsiStream(**stream_info)


  def _ping_device(self) -> None:
    return None


  def _connect(self) -> bool:
    try:
      TMSiSDK().discover(dev_type=DeviceType.saga,
                         dr_interface=DeviceInterfaceType.docked,
                         ds_interface=DeviceInterfaceType.usb)
      discovered_devices: list[TMSiDevice] = TMSiSDK().get_device_list(DeviceType.saga)
      if discovered_devices:
        # Get the handle to the first discovered device and open the connection.
        for device in discovered_devices:
          if device.get_dr_interface() == DeviceInterfaceType.docked:
            # Open the connection to SAGA.
            self.device = device
            self.device.open()
            break

        # Check the current bandwidth that's in use.
        current_bandwidth = self.device.get_device_bandwidth()
        print('The current bandwidth in use is {:} bit/s'.format(current_bandwidth['in use']), flush=True)
        print('Maximum bandwidth for wifi measurements is {:} bit/s'.format(current_bandwidth['wifi']), flush=True)

        # Maximal allowable sample rate with all enabled channels is 1000Hz.
        self.device.set_device_sampling_config(base_sample_rate=SagaBaseSampleRate.Decimal,
                                               channel_type=ChannelType.all_types,
                                               channel_divider=4)
        # NOTE: must match the hardcoded specs else wrong sensors will be read out.
        # channels
        # oxy goes to digi
        # breath to aux 1
        # gsr aux 2
        # double bip to bipolar
        # 65 66 double bipolar
        # 69 breath
        # 72 gsr
        # 78 blood oxy
        # 79, 80, 81, 82, 83, 84, 85, 86 -> sensors
        activated_channels = [65, 66, 69, 72, 78, 79, 80, 81, 82, 83, 84, 85, 86]
        self.device.set_device_active_channels(list(range(90)), False)
        self.device.set_device_active_channels(activated_channels, True)

        # Check the current bandwidth that's in use.
        current_bandwidth = self.device.get_device_bandwidth()
        print('The current bandwidth in use is {:} bit/s'.format(current_bandwidth['in use']), flush=True)

        # Choose the desired DR-DS interface type.
        self.device.set_device_interface(DeviceInterfaceType.wifi)
        
        # Close the connection to the device (with the original interface type).
        self.device.close()
        
      time.sleep(3)
      print('wifi setup starting', flush=True)
      
      # Connection over wifi.
      TMSiSDK().discover(dev_type=DeviceType.saga,
                         dr_interface=DeviceInterfaceType.wifi,
                         ds_interface=DeviceInterfaceType.usb,
                         num_retries=10)
      discovered_devices: list[TMSiDevice] = TMSiSDK().get_device_list(DeviceType.saga)
      if discovered_devices:
        # Get the handle to the first discovered device and open the connection.
        for device in discovered_devices:
          if device.get_dr_interface() == DeviceInterfaceType.wifi:
            # Open the connection to SAGA
            self.device = device
            self.device.open()
            break

        self.data_sampling_server = SampleDataServer()
        self.data_queue = queue.Queue()
        self.data_sampling_server.register_consumer(self.device.get_id(), self.data_queue)

        print("SAGA",'Successfully connected to the TMSi streamer.', flush=True)
        self.device.start_measurement(MeasurementType.SAGA_SIGNAL)
        return True
      return False
    except Exception as e:
      print(e)
      print("SAGA",'Unsuccessful connection to the TMSi streamer.', flush=True)
      return False


  def _keep_samples(self) -> None:
    # Clear the buffer queue of accumulated values during the system bring-up.
    try:
      while True:
        self.data_queue.get_nowait()
    except queue.Empty:
      return


  def _process_data(self) -> None:
    try:
      new_data: SampleData = self.data_queue.get(timeout=10.0)
      process_time_s = get_time()
      sample_block = np.array(array_to_matrix(new_data.samples, new_data.num_samples_per_sample_set))
      tag: str = "%s.data" % self._log_source_tag()
      for sample in sample_block.T:
        data = {
          'BIP-01': sample[0],
          'BIP-02': sample[1],
          'breath': sample[2],
          'GSR': sample[3],
          'SPO2': sample[4],
          'counter': sample[-1],
        }
        self._publish(tag=tag, process_time_s=process_time_s, data={'tmsi-data': data})
    except queue.Empty:
      if not self._is_continue_capture:
        self._send_end_packet()


  def _stop_new_data(self):
    self.device.stop_measurement()


  def _cleanup(self) -> None:
    # Set the DR-DS interface type back to docked
    self.device.set_device_interface(DeviceInterfaceType.docked)
    self.device.close()
    super()._cleanup()
