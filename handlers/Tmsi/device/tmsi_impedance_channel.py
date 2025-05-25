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
 * @file tmsi_impedance_channel.py 
 * @brief 
 * TMSi Impedance Channel interface.
 */


'''

class TMSiImpedanceChannel:
    """Class to handle the Apex impedance channel"""
    def __init__(self):
        """Initialize the impedance channel."""

    def get_channel_index(self):
        """Get the index of the channel.

        :return: index of the channel.
        :rtype: int
        """
        return self._channel_index

    def get_channel_name(self):
        """Get the name of the channel.

        :return: name of the channel.
        :rtype: str
        """
        return self._channel_name