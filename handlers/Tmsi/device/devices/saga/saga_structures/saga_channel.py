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
 * @file saga_channel.py 
 * @brief 
 * SAGA Channel object.
 */


'''

from ....tmsi_channel import TMSiChannel, ChannelType

class SagaChannel(TMSiChannel):
    """A class to handle Saga channels."""
    def __init__(self):
        """Initialize Saga channel"""
        super().__init__()
        self._sensor = None

    def get_sensor_information(self):
        """Get sensor information

        :return: sensor information
        :rtype: SagaSensor
        """
        return self._sensor

    def set_channel_information(self, channel_description):
        """Set the information of the channel.

        :param channel_description: channel description 
        :type channel_description: list[TMSiDevChDesc]
        """
        self._type = ChannelType(channel_description.ChannelType)
        self._format = channel_description.ChannelFormat
        self._chan_bandwidth = channel_description.ChannelBandWidth
        self._chan_divider = channel_description.ChanDivider
        self._enabled = self._chan_divider != -1
        self._imp_divider = channel_description.ImpDivider
        self._exp = channel_description.Exp
        self._unit_name = channel_description.UnitName.decode('windows-1252')
        self._def_name = channel_description.DefChanName.decode('windows-1252')
        if self._type == ChannelType.AUX:
            self._alt_name = self._def_name
        else:
            self._alt_name = channel_description.AltChanName.decode('windows-1252')

    def set_sensor_information(self, sensor, bipolar = False):
        """Set sensor information on Saga channel

        :param sensor: sensor
        :type sensor: SagaSensor
        :param bipolar: tells if the sensor is on a bipolar channel, optional, default False
        :type bipolar: bool
        """
        self._sensor = sensor
        if not bipolar:
            self._exp = 0
            self._alt_name = sensor.get_sensor_name()
            self._unit_name = sensor.get_sensor_unit_name()