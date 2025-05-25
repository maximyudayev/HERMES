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
 * @file apex_info.py 
 * @brief 
 * APEX Information object.
 */


'''

from ....tmsi_device_enums import *

from ..apex_API_enums import TMSiPairingStatus

from .apex_const import ApexConst

class ApexInfo():
    """Class to handle the information of the Apex.
    """
    def __init__(self, 
        dongle_serial_number = 0,
        serial_number = 0,
        id = ApexConst.TMSI_DEVICE_ID_NONE, 
        dr_interface = DeviceInterfaceType.none,
        pairing_status = PairingStatus.no_pairing_needed):
        """Initialize the information of Apex.

        :param dongle_serial_number: serial number of the dongle, defaults to 0
        :type dongle_serial_number: int, optional
        :param serial_number: serial number of the device, defaults to 0
        :type serial_number: int, optional
        :param id: id of the Apex, defaults to ApexConst.TMSI_DEVICE_ID_NONE
        :type id: int, optional
        :param dr_interface: interface with the Apex, defaults to DeviceInterfaceType.none
        :type dr_interface: _type_, optional
        :param pairing_status: pairing status, defaults to PairingStatus.no_pairing_needed
        :type pairing_status: PairingStatus, optional
        """
        self.__device_name = "Unknown"
        self.__dr_interface = dr_interface
        self.__dr_serial_number = serial_number
        self.__id = id
        self.__dongle_id = ApexConst.TMSI_DONGLE_ID_NONE
        self.__state = DeviceState.disconnected
        self.__pairing_status = pairing_status
        self.__dongle_serial_number = dongle_serial_number
        self.__num_hw_channels = 0
        self.__num_cycling_states = 0
        self.__num_imp_channels = 0
        self.__num_channels = 0

    def get_dr_interface(self):
        """Get interface type of the Apex.

        :return: interface type of the Apex.
        :rtype: DeviceInterfaceType
        """
        return self.__dr_interface

    def get_id(self):
        """Get id of the Apex.

        :return: id
        :rtype: int
        """
        return self.__id

    def get_name(self):
        """Get the name of the Apex

        :return: name
        :rtype: str
        """
        return self.__device_name
    
    def get_num_channels(self):
        """Get the number of channels.

        :return: number of channels.
        :rtype: int
        """
        return self.__num_channels

    def get_num_impedance_channels(self):
        """Get the number of impedance channels.

        :return: number of impedance channels.
        :rtype: int
        """
        return self.__num_imp_channels

    def get_dongle_serial_number(self):
        """Get the serial number of the dongle.

        :return: serial number of the dongle.
        :rtype: int
        """
        return self.__dongle_serial_number

    def get_dr_serial_number(self):
        """Get the serial number of the Apex.

        :return: serial number of the Apex.
        :rtype: int
        """
        return self.__dr_serial_number

    def get_pairing_status(self):
        """Get the pairing status.

        :return: pairing status.
        :rtype: PairingStatus
        """
        return self.__pairing_status

    def get_state(self):
        """Get the state of the Apex.

        :return: state of the Apex.
        :rtype: DeviceState
        """
        return self.__state

    def set_device_info_report(self, device_info_report):
        """Set the device info report.

        :param device_info_report: info report from the Apex.
        :type device_info_report: TMSiDevInfoReport
        """
        self.__num_channels = device_info_report.NrOfChannels
        self.__num_hw_channels = device_info_report.NrOfHWChannels
        self.__num_imp_channels = device_info_report.NrOfImpChannels
        self.__num_cycling_states = device_info_report.NrOfCyclingStates
        self.__device_name = device_info_report.DeviceName.decode('windows-1252')
        
    def set_dongle_id(self, dongle_id):
        """Set the id of the dongle.

        :param dongle_id: dongle id.
        :type dongle_id: int
        """
        self.__dongle_id = dongle_id

    def set_dongle_serial_number(self, dongle_serial_number):
        """Set the serial number of the dongle.

        :param dongle_serial_number: serial number of the dongle.
        :type dongle_serial_number: int
        """
        self.__dongle_serial_number = dongle_serial_number

    def set_dr_interface(self, dr_interface):
        """Set the interface of the Apex.

        :param dr_interface: interface of the Apex.
        :type dr_interface: DeviceInterface
        """
        self.__dr_interface = dr_interface

    def set_dr_serial_number(self, dr_serial_number):
        """Set the serial number of the Apex.

        :param dr_serial_number: serial number.
        :type dr_serial_number: int
        """
        self.__dr_serial_number = dr_serial_number

    def set_id(self, id):
        """Set the id of the Apex.

        :param id: id of the Apex.
        :type id: int
        """
        self.__id = id

    def set_pairing_status(self, pairing_status):
        """Set the pairing status.

        :param pairing_status: pairing status.
        :type pairing_status: PairingStatus
        """
        self.__pairing_status = TMSiPairingStatus(pairing_status)

    def set_state(self, state):
        """Set the state of the device.

        :param state: state of the device.
        :type state: DeviceState
        """
        self.__state = state
