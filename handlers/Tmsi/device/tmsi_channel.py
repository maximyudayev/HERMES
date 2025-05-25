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
 * @file tmsi_channel.py 
 * @brief 
 * TMSi Channel interface.
 */


'''

from .tmsi_device_enums import ChannelType


class TMSiChannel:
    """Class to handle the interface of TMSi channel.
    """
    def __init__(self):
        """Initialize the TMSi channel.
        """
        self._alt_name = "-"
        self._chan_bandwidth = 0
        self._chan_divider = -1
        self._def_name = "-"
        self._enabled = False
        self._exp = 0
        self._format = 0
        self._is_reference = 0
        self._imp_divider = -1
        self._index = -1
        self._sample_rate = 0
        self._type = ChannelType.unknown
        self._unit_name = "-"

    def get_channel_bandwidth(self):
        """Get channel bandwidth

        :return: bandwidth of the channel
        :rtype: int
        """
        return self._chan_bandwidth
    
    def get_channel_divider(self):
        """Get channel divider.

        :return: the divider of the channel over base sample rate.
        :rtype: int
        """
        return self._chan_divider

    def get_channel_exp(self):
        """Get channel exponential.

        :return: the exponential in base 10 of the channel.
        :rtype: int
        """
        return self._exp

    def get_channel_format(self):
        """Get channel format.

        :return: format of the channel.
        :rtype: int
        """
        return self._format

    def get_channel_imp_divider(self):
        """Get channel impedance divider.

        :return: the impedance divider of the channel over base sample rate.
        :rtype: int
        """
        return self._imp_divider

    def get_channel_index(self):
        """Get channel index.

        :return: the index of the channel.
        :rtype: int
        """
        return self._index

    def get_channel_name(self):
        """Get channel name.

        :return: name of the channel.
        :rtype: str
        """
        return self._alt_name

    def get_channel_sampling_frequency(self):
        """Get channel sampling frequency

        :return: sampling frequency for the channel
        :rtype: int
        """
        return self._sample_rate
    
    def get_channel_type(self):
        """Get channel type.

        :return: channel type.
        :rtype: ChannelType
        """
        return self._type

    def get_channel_unit_name(self):
        """Get the name of the unit of measurement of the channel.

        :return: unit of measurement.
        :rtype: str
        """
        return self._unit_name
    
    def is_reference(self):
        """Get if the channel is a reference channel.

        :return: 1 if it is reference, 0 if it is not reference.
        :rtype: int
        """
        if self._is_reference == 1:
            return 1
        else:
            return 0

    def set_channel_information(self):
        """Set the information of the channel.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this channel')

    def set_channel_divider(self, base_sample_rate, divider = -1):
        """Set channel divider

        :param base_sample_rate: base sample rate of the device
        :type base_sample_rate: int
        :param divider: divider of the base sample rate, defaults to -1
        :type divider: int, optional
        """
        self._chan_divider = int(divider)
        if divider > -1:
            self._sample_rate = int(base_sample_rate) // (2 ** divider)
    
    def set_channel_index(self, index):
        """Set channel index

        :param index: index of the channel
        :type index: int
        """
        self._index = index
    
    def set_channel_name(self, default_channel_name = None, alternative_channel_name = None):
        """Set the names of the channel

        :param default_channel_name: default name, defaults to None
        :type default_channel_name: str, optional
        :param alternative_channel_name: alternative name, defaults to None
        :type alternative_channel_name: str, optional
        """
        if default_channel_name is not None:
            self._def_name = default_channel_name
        if alternative_channel_name is not None:
            self._alt_name = alternative_channel_name

    def set_reference(self, is_reference):
        """Set the channel to reference.

        :param is_reference: 1 if it is reference, 0 if it is not.
        :type is_reference: int
        """
        self._is_reference = is_reference