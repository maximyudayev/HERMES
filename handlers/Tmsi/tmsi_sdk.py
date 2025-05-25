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
 * @file tmsi_sdk.py 
 * @brief 
 * Singleton class which handles the discovery of TMSi devices.
 */


'''

from .tmsi_utilities.singleton import Singleton
from .tmsi_utilities.tmsi_logger import TMSiLoggerActivity
from .device.tmsi_device_enums import *
from .device.devices.apex.apex_device import ApexDevice
from .device.devices.saga.saga_device import SagaDevice


class TMSiSDK(metaclass = Singleton):
    """Singleton class which handles the discovery of TMSi devices"""
    def __init__(self):
        """Initializes the object."""
        self.__apex_device_list = []
        self.__apex_dongle_list = []
        self.__saga_device_list = []
        
    def discover(self, 
        dev_type, 
        dr_interface = DeviceInterfaceType.none, 
        ds_interface = DeviceInterfaceType.none,
        num_retries = 3) -> tuple:
        """Discovers if there are available devices.

        :param dev_type: device type to search
        :type dev_type: DeviceType
        :param dr_interface: datarecorder interface, defaults to DeviceInterfaceType.none. See :meth:`SagaDevice.discover() <TMSiSDK.device.devices.saga.saga_device.SagaDevice.discover>` and :meth:`ApexDevice.discover() <TMSiSDK.device.devices.apex.apex_device.ApexDevice.discover>` for more details.
        :type dr_interface: DeviceInterfaceType, optional
        :param ds_interface: docking station interface (if needed), defaults to DeviceInterfaceType.none. See :meth:`SagaDevice.discover() <TMSiSDK.device.devices.saga.saga_device.SagaDevice.discover>` for more details.
        :type ds_interface: DeviceInterfaceType, optional
        :param num_retries: number of retry if nothing found
        :type num_retries: int, optional
        :return: list of devices and list of dongles
        :rtype: tuple[list[TMSiDevice], list[TMSiDongle]]
        """
        if dev_type == DeviceType.apex:
            ApexDevice.discover(dr_interface, num_retries)
            TMSiLoggerActivity().log("TMSi-SDK->>APEX-SDK: discover devices")
            self.__apex_device_list = ApexDevice.get_device_list(dr_interface)
            self.__apex_dongle_list = ApexDevice.get_dongle_list()
            return (self.__apex_device_list, self.__apex_dongle_list)
        elif dev_type == DeviceType.saga:
            SagaDevice.discover(dr_interface, ds_interface, num_retries)
            TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: discover devices")
            self.__saga_device_list = SagaDevice.get_device_list()
            return (self.__saga_device_list, [])
    
    def get_device_list(self, dev_type) -> list:
        """Gets the list of available devices.

        :param dev_type: device to get
        :type dev_type: DeviceType
        :return: list of available devices.
        :rtype: list[TMSiDevice]
        """
        if dev_type == DeviceType.apex:
            return self.__apex_device_list
        elif dev_type == DeviceType.saga:
            return self.__saga_device_list
        return []

    def get_dongle_list(self, dev_type) -> list:
        """Gets the list of available dongles.

        :param dev_type: device type dongle to get
        :type dev_type: DeviceType
        :return: list of available dongles.
        :rtype: list[TMSiDongle]
        """
        if dev_type == DeviceType.apex:
            return self.__apex_dongle_list
        return []
        

    def get_driver_version(self, dev_type) -> tuple:
        """Gets the driver version

        :param dev_type: the device type for the drivers
        :type dev_type: DeviceType
        :return: driver versions
        :rtype: tuple[dll version, usb driver version]
        """
        if dev_type == DeviceType.apex:
            version = ApexDevice.get_driver_version()
            TMSiLoggerActivity().log("TMSi-SDK->>APEX-SDK: get driver version")
            dll_version = "".join([chr(i) for i in version.DllVersionString if i != 0])
            usb_version = "".join([chr(i) for i in version.LibUsbVersionString if i != 0])
            return (dll_version, usb_version)
        
            
