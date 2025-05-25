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
 * @file tmsi_measurement.py 
 * @brief 
 * Measurement interface.
 */


'''

from ctypes import *
import queue
import time

from ..tmsi_utilities.decorators import LogPerformances
from ..tmsi_utilities.tmsi_logger import TMSiLogger as logger
from ..tmsi_utilities.tmsi_logger import TMSiLoggerActivity
from .tmsi_thread import TMSiThread


class TMSiMeasurement():
    """A class to handle all the TMSi Measurements.
    """
    def __init__(self, dev, name = "TMSi Measurement"):
        """Initialize the measurement.

        :param dev: Device to measure from.
        :type dev: TMSiDevice
        :param name: name of the measurement, defaults to "TMSi Measurement"
        :type name: str, optional
        """
        self._dev = dev
        self._name = name
        self._downloaded_samples = 0
        self._download_percentage = 0
        self._download_samples_limit = None
        self._sample_data_buffer_size = 409600
        self._retrieved_sample_sets = (c_uint)(0)
        self._retrieved_data_type = (c_int)(0)
        __MAX_SIZE_CONVERSION_QUEUE = 50
        self._conversion_queue = queue.Queue(__MAX_SIZE_CONVERSION_QUEUE)
        self._empty_read_counter = 0
        self._tic_timeout = None
        self._timeout = 3
        self._disable_live_impedance = 0
        self._disable_average_reference_calculation = 0
        self._float_channels = []
        self._sensor_channels = []
        self._basic_conversion={}
        TMSiLoggerActivity().log("TMSi-SDK->>{}: create measurement".format(self.get_name()))
        TMSiLoggerActivity().log("{}->>{}-SDK: GET device channels request".format(self.get_name(), self._dev.get_device_type()))
        channels = self._dev.get_device_active_channels()
        TMSiLoggerActivity().log("{}-SDK->>{}: GET device channels response".format(self._dev.get_device_type(), self.get_name()))
        for i in range(len(channels)):
            if channels[i].get_channel_format() == 0x0020 or channels[i].get_channel_format() == 0x0120:
                self._float_channels.append(i)
            else:
                if (channels[i].get_channel_type().value == 3) and (channels[i].get_sensor_information() != None):
                    self._sensor_channels.append(i)
                else:
                    conversion_factor=10**channels[i].get_channel_exp()
                    if not conversion_factor in self._basic_conversion:
                        self._basic_conversion[conversion_factor]=[i]
                    else:
                        self._basic_conversion[conversion_factor].append(i)
        self._sampling_thread = TMSiThread(
            name="Sampling Thread",
            looping_function = self._sampling_function,
            pause = 0.01)
        self._conversion_thread = TMSiThread(
            name="Conversion Thread",
            looping_function = self._conversion_function,
            pause = 0.01
        )

    @LogPerformances
    def get_conversion_pause(self):
        """Get the conversion pause.

        :return: the pause between each loop of the conversion thread.
        :rtype: float
        """
        return self._conversion_thread.get_pause()

    @LogPerformances
    def get_sampling_pause(self):
        """Get the sampling pause.

        :return: the pause between each loop of the sampling thread.
        :rtype: float
        """
        return self._sampling_thread.get_pause()

    @LogPerformances
    def is_timeout(self):
        """Check if the measurement is in timeout.

        :return: True if it is in timeout.
        :rtype: bool
        """
        if self._tic_timeout is None:
            return False
        return self._timeout < time.perf_counter() - self._tic_timeout

    @LogPerformances
    def get_download_percentage(self):
        """Get download percentage.

        :return: percentage of the download.
        :rtype: float
        """
        return self._download_percentage

    def get_name(self):
        """Get the name of the measurement.

        :return: name of the measurement.
        :rtype: str
        """
        return self._name
    
    @LogPerformances
    def set_conversion_pause(self, pause):
        """Set the conversion thread pause.

        :param pause: the pause between each loop of the conversion thread.
        :type pause: float
        """
        logger().debug("conversion pause set to {}".format(pause))
        self._conversion_thread.set_pause(pause)

    @LogPerformances
    def set_download_samples_limit(self, max_number_of_samples = None):
        """Set the limit to the samples to download.

        :param max_number_of_samples: max number of samples to download, defaults to None
        :type max_number_of_samples: int, optional
        """
        self._download_samples_limit = max_number_of_samples - 1

    @LogPerformances
    def set_sampling_pause(self, pause):
        """Set the sampling thread pause.

        :param pause: the pause between each loop of the sampling thread.
        :type pause: float
        """
        logger().debug("sampling pause set to {}".format(pause))
        self._sampling_thread.set_pause(pause)

    @LogPerformances
    def start(self):
        """Start the measurement.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this measurement')

    @LogPerformances
    def stop(self):
        """Stop the measurement.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this measurement')

    @LogPerformances
    def _conversion_function(self):
        raise NotImplementedError('method not available for this measurement')

    @LogPerformances
    def _sampling_function(self):
        raise NotImplementedError('method not available for this measurement')