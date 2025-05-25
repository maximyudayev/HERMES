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
 * @file apex_dongle.py 
 * @brief 
 * APEX Dongle object.
 */


'''

from ...tmsi_dongle import TMSiDongle

class ApexDongle(TMSiDongle):
    """A class to represent and handle the Apex Dongle"""
    def __init__(self, dongle_id:int, serial_number:int):
        """Initialize the dongle.

        :param dongle_id: id of the dongle.
        :type dongle_id: int
        :param serial_number: serial number of the dongle.
        :type serial_number: int
        """
        super().__init__(dongle_id, serial_number)