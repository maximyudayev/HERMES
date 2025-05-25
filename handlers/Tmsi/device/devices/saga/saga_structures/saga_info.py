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
 * @file saga_info.py 
 * @brief 
 * SAGA Information object.
 */


'''

from ....tmsi_device_enums import *

from .saga_const import SagaConst

class SagaInfo():
    """Class to handle the information of the Saga.
    """
    def __init__(self, 
        id = SagaConst.TMSI_DEVICE_ID_NONE, 
        dr_serial_number = 0,
        ds_serial_number = 0,
        dr_interface = DeviceInterfaceType.none,
        ds_interface = DeviceInterfaceType.none):
        
        self.__device_name = "Unknown"
        self.__ds_interface = ds_interface
        self.__ds_serial_number = ds_serial_number
        self.__dr_interface = dr_interface
        self.__dr_serial_number = dr_serial_number
        self.__id = id
        self.__state = DeviceState.disconnected
        self.__num_hw_channels = 0
        self.__num_channels = 0
        self.__num_active_channels = 0
        self.__num_active_imp_channels = 0
        self.__num_batteries = 0
        self.__num_sensors = 0
        self.__interface_bandwidth = 0
        self.__available_recordings = 0

    def get_available_recordings(self):
        """Get available recordings of the Saga.

        :return: available recordings of the Saga.
        :rtype: int
        """
        return self.__available_recordings

    def get_dr_interface(self):
        """Get dr interface type of the Saga.

        :return: dr interface type of the Saga.
        :rtype: DeviceInterfaceType
        """
        return self.__dr_interface

    def get_dr_serial_number(self):
        """Get the serial number of the dr.

        :return: serial number of the dr.
        :rtype: int
        """
        return self.__dr_serial_number

    def get_ds_interface(self):
        """Get ds interface type of the Saga.

        :return: ds interface type of the Saga.
        :rtype: DeviceInterfaceType
        """
        return self.__ds_interface

    def get_ds_serial_number(self):
        """Get the serial number of the ds.

        :return: serial number of the ds.
        :rtype: int
        """
        return self.__ds_serial_number

    def get_id(self):
        """Get id of the Saga.

        :return: id
        :rtype: int
        """
        return self.__id

    def get_interface_bandwidth(self):
        """Get interface bandwidth

        :return: interface bandwidth
        :rtype: int
        """
        return self.__interface_bandwidth
    
    def get_name(self):
        """Get the name of the Saga

        :return: name
        :rtype: str
        """
        return self.__device_name
    
    def get_num_active_channels(self):
        """Get the number of active channels.

        :return: number of active channels.
        :rtype: int
        """
        return self.__num_active_channels
    
    def get_num_active_imp_channels(self):
        """Get the number of active impedance channels.

        :return: number of active impedance channels.
        :rtype: int
        """
        return self.__num_active_imp_channels

    def get_num_channels(self):
        """Get the number of channels.

        :return: number of channels.
        :rtype: int
        """
        return self.__num_channels

    def get_num_sensors(self):
        """Get the number of sensors.

        :return: number of sensors.
        :rtype: int
        """
        return self.__num_sensors

    def get_state(self):
        """Get the state of the Saga.

        :return: state of the Saga.
        :rtype: DeviceState
        """
        return self.__state

    def set_device_status_report(self, device_status_report):
        """Set the device status report.

        :param device_status_report: status report from the Saga.
        :type device_status_report: TMSiDevStatReport
        """
        self.__num_channels = device_status_report.NrOfChannels
        self.__num_batteries = device_status_report.NrOfBatteries
                
    def set_device_config(self, device_config):
        """Set device configuration

        :param device_config: configuration of the device.
        :type device_config: TMSiDevGetConfig
        """
        self.__num_hw_channels = device_config.NrOfHWChannels
        self.__num_sensors = device_config.NrOfSensors
        self.__interface_bandwidth = device_config.InterFaceBandWidth * 1_000_000
        self.__available_recordings = device_config.AvailableRecordings
        self.__device_name = device_config.DeviceName.decode("windows-1252")
    
    def set_dr_interface(self, dr_interface):
        """Set the dr interface of the dr.

        :param dr_interface: dr interface of the dr.
        :type dr_interface: DeviceInterface
        """
        self.__dr_interface = dr_interface

    def set_dr_serial_number(self, dr_serial_number):
        """Set the serial number of the dr.

        :param dr_serial_number: serial number.
        :type dr_serial_number: int
        """
        self.__dr_serial_number = dr_serial_number

    def set_ds_interface(self, ds_interface):
        """Set the ds interface of the ds.

        :param ds_interface: ds interface of the Saga.
        :type ds_interface: DeviceInterface
        """
        self.__ds_interface = ds_interface

    def set_ds_serial_number(self, ds_serial_number):
        """Set the serial number of the ds.

        :param dr_serial_number: serial number.
        :type dr_serial_number: int
        """
        self.__ds_serial_number = ds_serial_number

    def set_id(self, id):
        """Set the id of the Saga.

        :param id: id of the Saga.
        :type id: int
        """
        self.__id = id

    def set_num_active_channels(self, num_active_channels:int):
        """Set the number of active channels

        :param num_active_channels: number of active channels
        :type num_active_channels: int
        """
        self.__num_active_channels = num_active_channels

    def set_num_active_imp_channels(self, num_active_imp_channels:int):
        """Set the number of active channels

        :param num_active_imp_channels: number of active impedance channels
        :type num_active_imp_channels: int
        """
        self.__num_active_imp_channels = num_active_imp_channels

    def set_state(self, state):
        """Set the state of the device.

        :param state: state of the device.
        :type state: DeviceState
        """
        self.__state = state
