'''
(c) 2023-2024 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file download_measurement.py 
 * @brief 
 * Class to handle the download of a file from the device.
 */


'''

from .....tmsi_utilities.decorators import LogPerformances
from .....tmsi_utilities.tmsi_logger import TMSiLoggerActivity
from ..apex_API_structures import TMSiDevSetCardFileReq
from ..apex_API_enums import SampleControl
from .signal_measurement import SignalMeasurement


class DownloadMeasurement(SignalMeasurement):
    """Class to handle the download measurements."""
    def __init__(self, dev, file_id: int, n_of_samples: int = None, name:str = "Download Measurement"):
        """Initialize the download measurement.

        :param dev: Device to download from.
        :type dev: TMSiDevice
        :param file_id: id of the file to download.
        :type file_id: int
        :param n_of_samples: number of samples to download, defaults to None
        :type n_of_samples: int, optional
        :param name: name of the measurement, defaults to "Download Measurement"
        :type name: str, optional
        """
        super().__init__(dev, name)
        self._file_id = file_id
        if n_of_samples is None:
            header, metadata = self._dev.get_device_card_file_info(self._file_id)
            self._n_of_samples = metadata.NumberOfSamples
        else:
            self._n_of_samples = n_of_samples
        self.set_download_samples_limit(self._n_of_samples)

    @LogPerformances
    def start(self):
        """Start the measurement.
        """
        self._dev.reset_device_data_buffer()
        file_request = TMSiDevSetCardFileReq()
        file_request.RecFileID = self._file_id
        file_request.StartCounter = 0
        file_request.NumberOfSamples = self._n_of_samples
        file_request.StartStop = SampleControl.StartSampling.value
        TMSiLoggerActivity().log("{}->>APEX-SDK: set device download request ON".format(self.get_name()))
        self._dev.set_device_download_file_request(file_request)
        TMSiLoggerActivity().log("{}->>Sampling Thread: start".format(self.get_name()))
        self._sampling_thread.start()
        TMSiLoggerActivity().log("{}->>Conversion Thread: start".format(self.get_name()))
        self._conversion_thread.start()

    @LogPerformances
    def stop(self):
        """Stop the measurement.
        """
        file_request = TMSiDevSetCardFileReq()
        file_request.RecFileID = self._file_id
        file_request.StartStop = SampleControl.StopSampling.value
        self._sampling_thread.stop()
        self._sampling_thread.join()
        TMSiLoggerActivity().log("{}->>Sampling Thread: stop".format(self.get_name()))
        self._conversion_thread.stop()
        self._conversion_thread.join()
        TMSiLoggerActivity().log("{}->>Conversion Thread: stop".format(self.get_name()))
        TMSiLoggerActivity().log("{}->>APEX-SDK: set device download request OFF".format(self.get_name()))
        self._dev.set_device_download_file_request(file_request)