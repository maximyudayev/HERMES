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
 * @file tmsi_dongle.py 
 * @brief 
 * TMSi Dongle interface.
 */


'''

class TMSiDongle:
    """A class to handle all the TMSi dongles."""
    def __init__(self, dongle_id, serial_number):
        """Initialize the dongle.

        :param dongle_id: id
        :type dongle_id: int
        :param serial_number: serial number.
        :type serial_number: int
        """
        self._dongle_id = dongle_id
        self._serial_number = serial_number

    def get_id(self):
        """Get id of the dongle.

        :return: id of the dongle.
        :rtype: int
        """
        return self._dongle_id

    def get_serial_number(self):
        """Get serial number of the dongle.

        :return: serial number of the dongle.
        :rtype: int
        """
        return self._serial_number