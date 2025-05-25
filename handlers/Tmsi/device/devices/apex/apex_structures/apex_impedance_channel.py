'''
(c) 2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

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
 * @file apex_impedance_channel.py 
 * @brief 
 * APEX Impedance Channel object.
 */


'''

from ....tmsi_impedance_channel import TMSiImpedanceChannel

class ApexImpedanceChannel(TMSiImpedanceChannel):
    """Class to handle the Apex impedance channel"""
    def __init__(self, channel_metadata):
        """Initialize the channel.

        :param channel_metadata: channel metadata.
        :type channel_metadata: TMSiChannelMetadata
        """
        self.set_device_impedance_channel_metadata(channel_metadata)

    def get_channel_index(self):
        """Get the index of the channel.

        :return: index.
        :rtype: int
        """
        return self._channel_index
    
    def get_channel_unit_name(self):
        """Get the unit of measurement of the channel.

        :return: unit of the real and imaginary impedances.
        :rtype: tuple(str, str)
        """
        return (self._impedance_re_unit, self._impedance_im_unit)
    
    def set_device_impedance_channel_metadata(self, channel_metadata):
        """Set the impedance channel metadata.

        :param channel_metadata: channel metadata.
        :type channel_metadata: TMSiChannelMetadata
        """
        self._channel_index = channel_metadata.ChanIdx
        self._channel_name = channel_metadata.ChanName.decode()
        self._impedance_im_unit = channel_metadata.ImpedanceImUnit.decode()
        self._impedance_re_unit = channel_metadata.ImpedanceReUnit.decode()