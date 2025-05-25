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
 * @file saga_device.py 
 * @brief 
 * SAGA device object
 */


'''

from copy import copy
import os
import struct
import time

from ....tmsi_errors.error import TMSiError, TMSiErrorCode
from ....tmsi_utilities.decorators import LogPerformances
from ....tmsi_utilities.tmsi_logger import TMSiLoggerActivity, TMSiLogger

from ...tmsi_device import TMSiDevice
from ...tmsi_device_enums import *

from .saga_structures.saga_const import SagaConst
from .saga_structures.saga_info import SagaInfo
from .saga_structures.saga_config import SagaConfig
from .saga_structures.saga_channel import SagaChannel
from .saga_structures.saga_sensor import SagaSensor
from .saga_API_structures import *
from .saga_API_enums import *
from .saga_API_lookup_table import DeviceErrorLookupTable

#from .saga_API import *

_MAX_NUM_BATTERIES = 2

class SagaDevice(TMSiDevice):
    """A class to represent and handle the Saga"""
    
    __saga_sdk = None
    __MAX_NUM_DEVICES = 10
    __device_info_list = []
    __DEVICE_TYPE = "SAGA"

    @LogPerformances
    def __init__(self,
                id:int,
                dr_interface:DeviceInterfaceType,
                dr_serial_number:int,
                ds_interface:DeviceInterfaceType,
                ds_serial_number:int):
        """Initialize the Device

        :param id: index of the device
        :type id: int
        :param dr_interface: interface DR-DS
        :type dr_interface: DeviceInterfaceType
        :param dr_serial_number: serial number of the DR
        :type dr_serial_number: int
        :param ds_interface: interface DS-computer
        :type ds_interface: DeviceInterfaceType
        :param ds_serial_number: serial number of the DS
        :type ds_serial_number: int
        """
        from .saga_API import DeviceHandle
        self.__device_handle = DeviceHandle(0)
        self.__info = SagaInfo(
            id = id,
            dr_interface = dr_interface,
            dr_serial_number = dr_serial_number,
            ds_interface = ds_interface,
            ds_serial_number = ds_serial_number)
        self.__config = SagaConfig()

    @LogPerformances
    def apply_mask(self, n_channels, masks):
        """Apply a mask to the received channels

        :param n_channels: list of indices of channels
        :type n_channels: int
        :param masks: list of mask functions
        :type masks: function
        :raises TMSiError: TMSiErrorCode.api_invalid_command if wrogn arguments
        """
        if not isinstance(n_channels, list):
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command,
                            message = "n_channels must be a list of integers")
        if not isinstance(masks, list):
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command,
                            message = "masks must be a list of functions")
        self.__config.set_mask_info(channels = n_channels, functions = masks)
        if hasattr(self, "__measurement"):
            if hasattr(self.__measurement, "apply_mask"):
                self.__measurement.apply_mask(mask = self.__config.get_mask_info())
    
    @LogPerformances
    def close(self):
        """Closes the connection to the device.

        :raises TMSiError: TMSiErrorCode.device_error if impossible to close.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        from .saga_API import TMSiCloseDevice
        if (self.__info.get_state() != DeviceState.disconnected):
            self.__last_error_code = TMSiCloseDevice(self.__device_handle)
            self.__info.set_state(DeviceState.disconnected)
            if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
                return 
            else:
                raise TMSiError(
                    error_code = TMSiErrorCode.device_error, 
                    dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
        else:
            raise TMSiError(TMSiErrorCode.device_not_connected)

    @LogPerformances
    def discover(dr_interface: DeviceInterfaceType, ds_interface: DeviceInterfaceType, num_retries: int = 3):
        """Discovers available devices.

        :param dr_interface: dr device interface to be searched. Allowed interfaces: docked, wifi and optical.
        :type dr_interface: DeviceInterfaceType
        :param ds_interface: ds device interface to be searched. Allowed interfaces: usb and network.
        :type ds_interface: DeviceInterfaceType
        :param num_retries: Number of retries, optional
        :type num_reties: int
        """
        from .saga_API import SagaDllAvailable, SagaDllLocked, TMSiGetDeviceList
        if not SagaDllAvailable:
            TMSiLogger().warning("SAGA DLL not available.")
            raise TMSiError(error_code=TMSiErrorCode.missing_dll)
        if SagaDllLocked:
            TMSiLogger().warning("SAGA DLL already in use.")
            raise TMSiError(error_code=TMSiErrorCode.already_in_use_dll)
        SagaDevice.get_sdk() # if sdk already available, nothing happens, otherwise initialize
        for i in range (SagaDevice.__MAX_NUM_DEVICES):
            if SagaDevice.__device_info_list[i].get_dr_interface() == dr_interface:
                SagaDevice.__device_info_list[i] = SagaInfo()
        
        device_list = (TMSiDevList * SagaDevice.__MAX_NUM_DEVICES)()
        num_found_devices = (c_uint)(0)

        for i in range (SagaDevice.__MAX_NUM_DEVICES):
            device_list[i].TMSiDeviceID = SagaConst.TMSI_DEVICE_ID_NONE

        while num_retries > 0:
            ret = TMSiGetDeviceList(pointer(device_list), SagaDevice.__MAX_NUM_DEVICES, ds_interface.value, dr_interface.value)

            if (ret == TMSiDeviceRetVal.TMSI_OK):
                # Devices are found, update the local device list with the found result
                for i in range (SagaDevice.__MAX_NUM_DEVICES):
                    if (device_list[i].TMSiDeviceID != SagaConst.TMSI_DEVICE_ID_NONE):
                        for ii in range (SagaDevice.__MAX_NUM_DEVICES):
                            if (SagaDevice.__device_info_list[ii].get_id() == SagaConst.TMSI_DEVICE_ID_NONE):
                                SagaDevice.__device_info_list[ii].set_id(device_list[i].TMSiDeviceID)
                                SagaDevice.__device_info_list[ii].set_dr_interface(dr_interface)
                                SagaDevice.__device_info_list[ii].set_dr_serial_number(device_list[i].DRSerialNr)
                                SagaDevice.__device_info_list[ii].set_ds_interface(ds_interface)
                                SagaDevice.__device_info_list[ii].set_ds_serial_number(device_list[i].DSSerialNr)
                                SagaDevice.__device_info_list[ii].set_state(DeviceState.disconnected)

                                num_retries = 0
                                break
                num_retries -= 1
            else:
                num_retries -= 1
                TMSiLogger().warning('Impossible to find devices. Number of retries left: ' + str(num_retries))

    @LogPerformances
    def download_file_from_device(self, file_id: int, filename: str = None):
        """Creates a data stream to download the file from the device.

        :param file_id: id of the file to download.
        :type file_id: int
        :param filename: filename where to write the impedance report (if available), defaults to None
        :type filename: str, optional
        :raises TMSiError: TMSiErrorCode.file_writer_error if impedance report download fails.
        :raises TMSiError: TMSiErrorCode.device_error if the file download fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if not isinstance(file_id, int):
            raise TMSiError(
                error_code=TMSiErrorCode.api_incorrect_argument,
                message = "file requested does not exist.")
        file_info = self.get_device_card_file_info(file_id)
        n_of_samples = file_info["metadata"].NoOfSamples
        if n_of_samples < 1:
            raise TMSiError(
                error_code=TMSiErrorCode.api_incorrect_argument,
                message = "file requested does not exist.")
        self.start_download_file(file_id, filename, n_of_samples)
        self.__user_abort = False
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: start download")
        while True:
            percentage = self.__measurement.get_download_percentage()
            TMSiLogger().debug("downloaded: {:.2f}%".format(percentage))
            if percentage >= 100:
                break
            if self.__measurement.is_timeout():
                break
            if self.__user_abort:
                break
            time.sleep(0.1)

        self.stop_download_file()
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: stop download")

    @LogPerformances
    def export_configuration(self, filename: str):
        """Exports the current configuration to an xml file.

        :param filename: name of the file where to export configuration.
        :type filename: str
        :raises TMSiError: TMSiErrorCode.file_writer_error if export configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: export configuration")
        if self.__info.get_state() == DeviceState.connected:
            TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: export configuration")
            if self.__config.export_to_xml(filename):
                TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: export succeded")
                return
            else:
                TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: export failed file write error")
                raise TMSiError(error_code = TMSiErrorCode.file_writer_error)
        TMSiLoggerActivity().log("SAGA-SDK->>TMSi-SDK: export failed device not connected")
        raise TMSiError(
            error_code = TMSiErrorCode.device_not_connected)
    
    @LogPerformances
    def import_configuration(self, filename: str):
        """Imports the file configuration to the device.

        :param filename: name of the file where to export configuration.
        :type filename: str
        :raises TMSiError: TMSiErrorCode.general_error if import configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: import configuration")
        if self.__info.get_state() == DeviceState.connected:
            TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: import configuration")
            import_success, import_error = self.__config.import_from_xml(filename)
            if import_success:
                self.__set_device_config()
                TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: set device configuration")
                self.__load_config_from_device()
                TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get configuration")
                return
            else:
                TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: import failed general error")
                raise TMSiError(error_code = TMSiErrorCode.file_import_error, message = import_error)
        TMSiLoggerActivity().log("SAGA-SDK->>TMSi-SDK: import failed device not connected")
        raise TMSiError(
            error_code = TMSiErrorCode.device_not_connected)
    
    @LogPerformances
    def get_card_recording_config(self) -> TMSiDevRecCfg:
        """Gets configuration for card recording.

        :raises TMSiError: TMSiErrorCode.device_error if get configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: Configuration of the card recording.
        :rtype: TMSiDevRecCfg
        """
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get card recording configuration")
        if self.__info.get_state() == DeviceState.connected:
            return self.__get_device_card_recording_config()
        raise TMSiError(
            error_code = TMSiErrorCode.device_not_connected)

    @LogPerformances
    def get_card_status(self) -> SagaCardStatus:
        """Get card status

        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: saga card status
        :rtype: SagaCardStatus
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        card_status = SagaCardStatus()
        card_status.NrOfRecordings = self.__info.get_available_recordings()
        full_status = self.__get_full_device_status()
        card_status.TotalSpace = full_status["TotalSizeMB"]
        card_status.AvailableSpace = full_status["UsedSizeMB"]
        return card_status

    @LogPerformances
    def get_device_active_channels(self):
        """Gets the list of active channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[SagaChannel]
        """
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-SDK: get device active channels")
        self.get_device_channels()
        return self.__config.get_active_channels()
    
    @LogPerformances
    def get_device_active_impedance_channels(self):
        """Gets the list of active impedance channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[SagaChannel]
        """
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-SDK: get device active impedance channels")
        self.get_device_channels()
        return self.__config.get_active_imp_channels()
    
    @LogPerformances
    def get_device_bandwidth(self):
        """Get bandwidths of the device

        :return: Bandwidths of the device
        :rtype: dict
        """
        bandwidths = {}
        bw_requested = 0
        for channel in self.__config.get_active_channels():
            bw_requested += channel.get_channel_bandwidth()
        bandwidths["in use"] = bw_requested
        bandwidths["available"] = self.__info.get_interface_bandwidth()
        bandwidths["docked"] = 32_000_000
        bandwidths["internal memory"] = 2_000_000
        bandwidths["optical"] = 32_000_000
        bandwidths["wifi"] = 2_000_000
        return bandwidths
    
    @LogPerformances
    def get_device_card_file_info(self, file_id: int) -> dict:
        """Gets the information of the file on the device's card.

        :param file_id: Id of the file to be investigated.
        :type file_id: int
        :raises TMSiError: TMSiErrorCode.device_error if get card info fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: A dictionary with file metadata and impedance report.
        :rtype: {"metadata":TMSiDevRecDetails, "impedance_report": [TMSiDevImpReport]}
        """
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get card file info")
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        metadata, impedance_report = self.__get_device_card_file_metadata(file_id)
        file_info = {}
        file_info["metadata"] = metadata
        file_info["impedance_report"] = impedance_report
        return file_info

    @LogPerformances
    def get_device_card_file_list(self):
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get card file list")
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(
                error_code = TMSiErrorCode.device_not_connected)
        return self.__get_device_card_file_list()

    @LogPerformances
    def get_device_channels(self) -> list:
        """Gets the list of channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[SagaChannel]
        """
        TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: get device channels request")
        if self.__info.get_state() == DeviceState.sampling:
            TMSiLoggerActivity().log("SAGA-SDK->>SAGA-SDK: device is sampling, get device channel from configuration")
            return self.__config.get_channels()
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get device channels")
        self.__get_device_configuration()
        TMSiLoggerActivity().log("SAGA-SDK->>TMSi-SDK: get device channels response")
        return self.__config.get_channels()
    
    @LogPerformances
    def get_device_data(self, 
        POINTER_received_data_array: pointer, 
        buffer_size: int, 
        POINTER_num_of_sets: pointer, 
        POINTER_data_type: pointer) -> TMSiDeviceRetVal:
        """Gets data from the device during sampling.

        :param POINTER_received_data_array: array that will contain the received data.
        :type POINTER_received_data_array: pointer(array[c_float])
        :param buffer_size: maximum size of the buffer.
        :type buffer_size: int
        :param POINTER_num_of_sets: number of sets of data received.
        :type POINTER_num_of_sets: pointer(c_uint)
        :param POINTER_data_type: type of data received.
        :type POINTER_data_type: pointer(c_int)
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is notin  samplin modeg
        :return: return value of the call
        :rtype: TMSiDeviceRetVal
        """
        from .saga_API import TMSiGetDeviceData
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get device data")
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command)
        return TMSiGetDeviceData(
            self.__device_handle, 
            POINTER_received_data_array, 
            buffer_size, 
            POINTER_num_of_sets, 
            POINTER_data_type)

    @LogPerformances
    def get_device_handle_value(self) -> int:
        """Returns the value of the device handle.

        :return: Device handle.
        :rtype: int
        """
        return self.__device_handle.value

    @LogPerformances
    def get_device_impedance_channels(self) -> list:
        """Gets the list of impedance channels.

        :raises TMSiError: TMSiErrorCode.device_error if get channels from the device fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: The list of channels
        :rtype: list[SagaImpedanceChannel]
        """
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-SDK: get device impedance channel")
        return self.__config.get_active_imp_channels()
    
    def get_device_impedance_data(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_info_report(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_device_interface(self):
        """Get device interface

        :return: device interface
        :rtype: DeviceInterfaceType
        """
        return DeviceInterfaceType(self.__config.get_configured_interface())

    @LogPerformances
    def get_device_list() -> list:
        """Gets the list of available devices.

        :return: a list of available devices on the requested interface.
        :rtype: list[SagaDevice]
        """
        device_list = []
        for idx in range (SagaDevice.__MAX_NUM_DEVICES):
            if SagaDevice.__device_info_list[idx].get_id() != SagaConst.TMSI_DEVICE_ID_NONE:
                device_list.append(
                    SagaDevice(id = SagaDevice.__device_info_list[idx].get_id(),
                               dr_interface = SagaDevice.__device_info_list[idx].get_dr_interface(),
                               dr_serial_number = SagaDevice.__device_info_list[idx].get_dr_serial_number(),
                               ds_interface = SagaDevice.__device_info_list[idx].get_ds_interface(),
                               ds_serial_number = SagaDevice.__device_info_list[idx].get_ds_serial_number()))
        return device_list

    @LogPerformances
    def get_device_name(self):
        """Get name of the device

        :return: name of the device
        :rtype: str
        """
        return self.__info.get_name()

    def get_device_power_status(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_device_references(self):
        """Get device references

        :return: A dictionary whith reference and autoreference
        :rtype: dict
        """
        refs = dict()
        refs["reference"] = self.__config.get_reference_method()
        refs["autoreference"] = self.__config.get_auto_reference_method()
        return refs
    
    @LogPerformances
    def get_device_repair_logging(self):
        """Get device repair logging

        :return: return if repair logging is enabled or not
        :rtype: bool
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(TMSiErrorCode.device_not_connected)
        return self.__config.get_repair_logging() == 1

    def get_device_sampling_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_device_sampling_frequency(self, detailed = False) -> int:
        """Gets the sampling frequency.

        :param detailed: detailed frequency description, defaults to False
        :type detailed: bool, optional
        :return: sampling frequency or frequencies.
        :rtype: int or dictionary if detailed
        """
        if detailed:
            frequencies = dict()
            frequencies["device_sampling_rate"] = self.__config.get_sampling_frequency()
            frequencies["base_sampling_rate"] = self.__config.get_sample_rate()
            for channel in self.get_device_active_channels():
                channel.set_channel_divider(
                    divider = channel.get_channel_divider(), 
                    base_sample_rate = frequencies["base_sampling_rate"])
                field = "{}_sampling_frequency".format(str(channel.get_channel_type()).replace("ChannelType.",""))
                if field not in frequencies:
                    frequencies[field] = channel.get_channel_sampling_frequency()
            return frequencies
        return self.__config.get_sampling_frequency()

    @LogPerformances
    def get_device_serial_number(self) -> int:
        """Gets the serial number of the device.

        :return: serial number of the device.
        :rtype: int
        """
        return self.__info.get_dr_serial_number()
           
    @LogPerformances
    def get_device_sync_out_config(self):
        """Get device sync out configuration

        :return: information about the sync out
        :rtype: dict
        """
        sync_out = dict()
        sync_out["marker"] = self.__config.get_dr_sync_out_divider() == -1
        sync_out["frequency"] = int(self.__config.get_sample_rate() / self.__config.get_dr_sync_out_divider())
        sync_out["duty_cycle"] = self.__config.get_dr_sync_out_duty_cycle() / 10
        return sync_out
        
    @LogPerformances
    def get_device_state(self) -> DeviceState:
        """Gets the state of the device.

        :return: the device state.
        :rtype: DeviceState
        """
        return self.__info.get_state()
    
    def get_device_time(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_device_triggers(self):
        """Get device triggers

        :return: True if triggers enabled, False otherwise
        :rtype: boolean
        """
        return self.__config.get_triggers() == 1
    
    @LogPerformances
    def get_device_type(self) -> str:
        """Returns the device type.

        :return: the device type.
        :rtype: str
        """
        return SagaDevice.__DEVICE_TYPE

    def get_dongle_list(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_dongle_serial_number(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_downloaded_percentage(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_dr_interface(self) -> DeviceInterfaceType:
        """Returns the interface of the device.

        :return: the interface of the device.
        :rtype: DeviceInterfaceType
        """
        return self.__info.get_dr_interface()

    def get_event(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_event_buffer(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_file_channels(self, file_id):
        """Gets the list of channels from the file.

        :param file_id: id of the file
        :type file_id: int
        :return: The list of channels
        :rtype: list[ApexChannel]
        """
        return self.get_device_active_channels()
        
    @LogPerformances
    def get_id(self) -> int:
        """Gets the device id.

        :return: the device id.
        :rtype: int
        """
        return id(self)

    def get_live_impedance(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_num_active_channels(self) -> int:
        """Returns the number of active channels of the device.

        :return: number of active channels of the device.
        :rtype: int
        """
        return self.__info.get_num_active_channels()

    @LogPerformances
    def get_num_active_impedance_channels(self) -> int:
        """Returns the number of active impedance channels of the device.

        :return: number of active impedance channels of the device.
        :rtype: int
        """
        return self.__info.get_num_active_imp_channels()
    
    def get_num_channels(self) -> int:
        """Returns the number of channels of the device.

        :return: number of channels of the device.
        :rtype: int
        """
        return self.__info.get_num_channels()

    def get_num_impedance_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def get_sdk() -> CDLL:
        """Gets the handle of the communication library

        :return: the handle of the communication library.
        :rtype: CDLL
        """
        if SagaDevice.__saga_sdk is None:
            SagaDevice.__initialize()
        return SagaDevice.__saga_sdk
    
    @LogPerformances
    def open(self):
        """Opens the connection with the device.

        :raises TMSiError: TMSiErrorCode.device_error if get time from the device fails.
        :raises TMSiError: TMSiErrorCode.no_devices_found if device not found.
        """
        from .saga_API import TMSiOpenDevice, TMSiCloseDevice
        TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: open connection")
        if self.__info.get_id() != SagaConst.TMSI_DEVICE_ID_NONE:
            TMSiLoggerActivity().log("SAGA-SDK->>SAGA-DLL: open connection")
            self.__last_error_code = TMSiOpenDevice(
                pointer(self.__device_handle),
                self.__info.get_id(),
                self.__info.get_dr_interface().value
            )
            if self.__last_error_code == TMSiDeviceRetVal.TMSI_DS_DEVICE_ALREADY_OPEN:
                # The found device is available but in it's open-state: Close and re-open the connection
                self.__last_error_code = TMSiCloseDevice(
                    self.__device_handle)
                self.__last_error_code = TMSiOpenDevice(
                    pointer(self.__device_handle),
                    self.__info.get_id(),
                    self.__info.get_dr_interface().value
            )

            if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
                    TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: open connection succeeded")
                    # The device is opened succesfully. Update the device information.
                    self.__info.set_state(DeviceState.connected)
                    # Read the device's configuration
                    self.__load_config_from_device()

            else:
                TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: open connection failed, device error")
                raise TMSiError(
                    error_code = TMSiErrorCode.device_error, 
                    dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
        else:
            TMSiLoggerActivity().log("SAGA-SDK->>TMSi-SDK: open connection failed, no device found")
            raise TMSiError(
                error_code = TMSiErrorCode.no_devices_found)

    def pair_device(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def reload_device(self):
        """Reload information from device
        
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in connected state
        """
        if (self.__info.get_state() != DeviceState.connected):
            raise TMSiError(TMSiErrorCode.device_not_connected)
        self.__load_config_from_device()
    
    @LogPerformances
    def reset_device_card(self):
        """Resets the memory card of the device.

        :raises TMSiError: TMSiErrorCode.device_error if reset card fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: reset device card")
        self.__reset_device_card()
    
    @LogPerformances
    def reset_device_data_buffer(self):
        """Resets the incoming buffer.

        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is notin  samplin modeg
        :raises TMSiError: TMSiErrorCode.device_error if reset fails
        """
        from .saga_API import TMSiResetDeviceDataBuffer
        if (self.__info.get_state() != DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiResetDeviceDataBuffer(self.__device_handle)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    def reset_device_event_buffer(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def reset_masks(self):
        """Reset masks"""
        self.apply_mask(n_channels = [], masks = [])

    @LogPerformances
    def reset_to_factory_default(self):
        """Resets the device to default configuration. 
        To fully complete the progress, please :
        1. close the connection with the device, 
        2. undock your data recorder and remove batteries.

        :raises TMSiError: TMSiErrorCode.device_error if reset fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        TMSiLoggerActivity().log("TMSi-SDK->>SAGA-SDK: reset to factory")
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: reset to factory")
        self.__set_device_factory_default()
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get device configuration")
        self.__load_config_from_device()

    @LogPerformances
    def set_card_recording_config(self, config: TMSiDevRecCfg) -> TMSiDevRecCfg:
        """Sets the configuration for recording on card.

        :param config: configuration to be set.
        :type config: TMSiDevRecCfg
        :raises TMSiError: TMSiErrorCode.device_error if set configuration fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :return: the new available configuration for recording on card.
        :rtype: TMSiDevRecCfg
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: set device card recording configuration")
        self.__set_device_card_recording_config(config)
        TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: get device card recording configuration")
        return self.__get_device_card_recording_config()

    @LogPerformances
    def set_device_active_channels(self, indices, activate):
        """Set the activity of the channel.

        :param indices: list of indices to activate/deactivate
        :type indices: list[int]
        :param activate: True to activate, False to deactivate
        :type activate: bool
        """
        channels = self.__config.get_channels()
        dict_type = {}
        for channel in channels:
            if channel.get_channel_divider() == -1:
                continue
            if channel.get_channel_type() not in dict_type:
                dict_type[channel.get_channel_type()] = channel.get_channel_divider()
                continue
            if channel.get_channel_divider() < dict_type[channel.get_channel_type()]:
                dict_type[channel.get_channel_type()] = channel.get_channel_divider()

        for index in indices:
            if channels[index].get_channel_type() not in dict_type:
                dict_type[channels[index].get_channel_type()] = 0
            channels[index].set_channel_divider(
                divider = dict_type[channels[index].get_channel_type()] if activate else -1,
                base_sample_rate = self.__config.get_sample_rate())
        
        self.__set_device_config()
        self.__load_config_from_device()

    @LogPerformances
    def set_device_backup_logging(self, enable = True, prefix_filename : str = None):
        """Set the backup logging during acquisition

        :param enable: True if enabled, False otherwise, defaults to True
        :type enable: bool, optional
        :param prefix_filename: Prefix for the file name to save on the device card, defaults to None
        :type prefix_filename: str, optional
        """
        cfg = self.get_card_recording_config()
        if enable:
            cfg.StartControl = SagaStartCardRecording.Remote.value
        else:
            cfg.StartControl = SagaStartCardRecording.Off.value
        if prefix_filename is not None:
            max_len = len(prefix_filename)
            prefix_filename = bytearray(prefix_filename, 'utf-8')
            converted_str = bytearray(SagaStringLengths.PrefixFileName.value)
            converted_str[:max_len] = prefix_filename[:max_len]
            cfg.PrefixFileName[:] = converted_str
        self.set_card_recording_config(cfg)
    
    @LogPerformances
    def set_device_channel_names(self, 
        names: list, 
        indices: list) -> list:
        """Sets the device channel names

        :param names: names to be set.
        :type names: list[str]
        :param indices: index of the channels to edit.
        :type indices: list[int]
        :raises TMSiError: TMSiErrorCode.device_error if set names fails.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        :raises TypeError: if names is not only strings
        :raises TypeError: if indices is not only integers.
        :return: list of new channels.
        :rtype: list[SagaChannel]
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        for name in names:
            if not isinstance(name, str):
                raise TypeError("names must be strings")
        for index in indices:
            if not isinstance(index, int):
                raise TypeError("indices must be integers")
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: set device channel configuration")
        self.__set_device_channel_names(names, indices)
        self.__set_device_config()
        self.__load_config_from_device()

    @LogPerformances
    def set_device_download_file_request(self, file_id, download = True):
        """Sets the download file request to start or stop the download stream.

        :param file_id: id of the file to download
        :type file_id: int
        :param download: True if to download, False to stop, defaults to True
        :type download: bool, optional
        :raises TMSiError: TMSiErrorCode.device_error if set device download file request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        from .saga_API import TMSiGetRecordingFile
        recording_metadata = TMSiDevRecDetails()
        impedance_report_list = (TMSiDevImpReport*self.__info.get_num_active_imp_channels())()
        impedance_report_list_len = self.__info.get_num_active_imp_channels()
        self.__last_error_code = TMSiGetRecordingFile(
            self.__device_handle,
            file_id,
            SampleControl.STARTSamplingDevice.value if download else SampleControl.STOPSamplingDevice.value,
            pointer(recording_metadata),
            pointer(impedance_report_list),
            impedance_report_list_len)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            impedances = []
            for i in range(impedance_report_list_len):
                impedances.append(impedance_report_list[i])
            return recording_metadata, impedances
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def set_device_impedance_request(self, measurement_request: TMSiDevImpReq):
        """Sets the impedance request to start or stop the acquisition.

        :param measurement_request: measurement request to start or stop.
        :type measurement_request: TMSiSetDeviceImpedanceRequest
        :raises TMSiError: TMSiErrorCode.device_error if set impedance request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        from .saga_API import TMSiSetDeviceImpedance
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command)
        self.__last_error_code = TMSiSetDeviceImpedance(
            self.__device_handle,
            pointer(measurement_request)
        )
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def set_device_interface(self, device_interface):
        """Set device interface DS-DR

        :param device_interface: Interface between DS and DR
        :type device_interface: DeviceInterfaceType
        :raises ValueError: if interface is not of type DeviceInterfaceType
        :raises TMSiError: TMSiErrorCode.api_incompatible_configuration if interface is not allowed
        """
        if not isinstance(device_interface, DeviceInterfaceType):
            raise ValueError("device_interface must be DeviceInterfaceType class")
        allowed_device_interfaces = [
            DeviceInterfaceType.docked,
            DeviceInterfaceType.optical,
            DeviceInterfaceType.wifi]
        if device_interface not in allowed_device_interfaces:
            raise TMSiError(error_code = TMSiErrorCode.api_incompatible_configuration, 
                            message = "interface not allowed")
        self.__config.set_configured_interface(device_interface)
        self.__set_device_config()
        self.__load_config_from_device()
        self.__check_bandwidth()

    @LogPerformances
    def set_device_references(self, reference_method = None, auto_reference_method = None):
        """Set references for the device

        :param reference_method: reference method to apply to UNI channels, defaults to None
        :type reference_method: ReferenceMethod, optional
        :param auto_reference_method: reference method to switch to UNI channels if common mode is disconnected, defaults to None
        :type auto_reference_method: ReferenceSwitch, optional
        """
        if reference_method is not None:
            self.__config.set_device_reference_method(reference_method=reference_method)
        if auto_reference_method is not None:
            self.__config.set_device_auto_reference_method(auto_reference_method=auto_reference_method)
        self.__set_device_config()
        self.__load_config_from_device()

    @LogPerformances
    def set_device_repair_logging(self, enable_repair_logging = True):
        """Set device repair logging

        :param enable_repair_logging: enable or not the repair logging, defaults to True
        :type enable_repair_logging: bool, optional
        :raises TMSiError: device_not_connected if not in connected state.
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(TMSiErrorCode.device_not_connected)
        self.__config.set_repair_logging(enable_repair_logging = enable_repair_logging)    
        self.__set_device_config()
        self.__load_config_from_device()

    @LogPerformances
    def set_device_sampling_config(
        self, 
        base_sample_rate = None,
        channel_type = None,
        channel_divider = 1):
        """Set sampling configuration for the device

        :param base_sample_rate: Base sample rate, defaults to None
        :type base_sample_rate: SagaBaseSampleRate, optional
        :param channel_type: Channel type to set, defaults to None
        :type channel_type: ChannelType, optional
        :param channel_divider: channel divider to apply to this type of channels, defaults to 1. Possible numbers are 1, 2, 4 or 8
        :type channel_divider: int, optional
        """
        allowed_dividers = [-1, 1, 2, 4, 8]
        if channel_divider not in allowed_dividers:
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        channel_divider = allowed_dividers.index(channel_divider) - 1
        if channel_type is not None:
            self.__set_device_channel_sample_rates(channel_type, channel_divider)
        if base_sample_rate is not None:
            # when base sample rate changes, a check on the sync out pulse must be done.
            if self.__config.get_dr_sync_out_divider() != 0:
                sync_freq = self.__config.get_sample_rate() / self.__config.get_dr_sync_out_divider()
                self.__config.set_dr_sync_out_divider(round(base_sample_rate.value / sync_freq))
        self.__set_device_config(base_sample_rate = base_sample_rate)
        self.__load_config_from_device()

    @LogPerformances
    def set_device_sampling_request(self, measurement_request: TMSiDevSampleReq):
        """Sets the sampling request to start or stop the acquisition.

        :param measurement_request: measurement request to configure the acquisition.
        :type measurement_request: TMSiDevSampleRequest
        :raises TMSiError: TMSiErrorCode.device_error if set sampling request fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if device is not in sampling mode.
        """
        from .saga_API import TMSiSetDeviceSampling
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command)
        TMSiLoggerActivity().log("SAGA-SDK->>SAGA-API: set device sampling request")
        self.__last_error_code = TMSiSetDeviceSampling(
            self.__device_handle, 
            pointer(measurement_request))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return
        else:
            TMSiLoggerActivity().log("SAGA-API->>SAGA-SDK: start failed with error {}".format(self.__last_error_code))
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def set_device_sync_out_config(self, marker = False, frequency = None, duty_cycle = None):
        """Set configuration sync out

        :param marker: marker, defaults to False
        :type marker: bool, optional
        :param frequency: frequency of the sync out, defaults to None
        :type frequency: float, optional
        :param duty_cycle: duty cycle of the sync out, defaults to None
        :type duty_cycle: float, optional
        :raises TMSiError: TMSiErrorCode.device_not_connected if not in connected state
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        if marker:
            self.__config.set_dr_sync_out_divider(divider = -1)
        else:
            if frequency:
                self.__config.set_dr_sync_out_divider(round(self.__config.get_sample_rate() / frequency))
            if duty_cycle:
                self.__config.set_dr_sync_out_duty_cycle(duty_cycle = int(duty_cycle * 10))
        self.__set_device_config()
        self.__load_config_from_device()        
        
    @LogPerformances
    def set_device_triggers(self, triggers):
        """Set device triggers

        :param triggers: True if enabled, False otherwise
        :type triggers: bool
        :raises TMSiError: device_not_connected if not in connected state
        """
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        self.__config.set_triggers(triggers = triggers)
        self.__set_device_config()
        self.__load_config_from_device()
    
    def set_device_time(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    @LogPerformances
    def start_download_file(self, file_id: int, filename: str = None, n_of_samples: int = None):
        """Starts the download of the file requested.

        :param file_id: id of the file to download.
        :type file_id: int
        :param filename: filename where to write the impedance report (if available), defaults to None
        :type filename: str, optional
        :raises TMSiError: TMSiErrorCode.file_writer_error if impedance report download fails.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if (self.__info.get_state() == DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if (self.__info.get_state() != DeviceState.connected):
            raise TMSiError(TMSiErrorCode.device_not_connected)
        if filename is not None:
            self.__download_impedance_report(file_id, filename)
        self.__measurement = MeasurementType.SAGA_DOWNLOAD(self, file_id, n_of_samples)
        self.__info.set_state(DeviceState.sampling)
        TMSiLoggerActivity().log("TMSi-SDK->>{}: start".format(self.__measurement.get_name()))
        self.__measurement.start()
        
    @LogPerformances
    def start_measurement(self, measurement_type: MeasurementType, thread_refresh = None):
        """Starts the measurement requested.

        :param measurement_type: measurement to start
        :type measurement_type: MeasurementType
        :param thread_refresh: refresh time for sampling and conversion threads, defaults to None.
        :type thread_refresh: float, optional.
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if (self.__info.get_state() == DeviceState.sampling):
            raise TMSiError(TMSiErrorCode.api_invalid_command)
        if self.__info.get_state() != DeviceState.connected:
            raise TMSiError(error_code = TMSiErrorCode.device_not_connected)
        self.__check_bandwidth()
        self.__measurement = measurement_type(self)
        if thread_refresh is not None:
            self.__measurement.set_sampling_pause(thread_refresh)
            self.__measurement.set_conversion_pause(thread_refresh)
        self.__info.set_state(DeviceState.sampling)
        if hasattr(self.__measurement, "apply_mask"):
            self.__measurement.apply_mask(mask = self.__config.get_mask_info())
        TMSiLoggerActivity().log("TMSi-SDK->>{}: start".format(self.__measurement.get_name()))
        self.__measurement.start()
        
    @LogPerformances
    def stop_download_file(self):
        """Stops the download of the file.

        :raises TMSiError: TMSiErrorCode.api_invalid_command if not sampling.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command)
        TMSiLoggerActivity().log("TMSi-SDK->>{}: stop".format(self.__measurement.get_name()))
        try:
            self.__measurement.stop()
        except TMSiError as e:
            TMSiLogger().warning(message = "stop_download_file failed with error:\n{}\nDevice state moved to connected to allow the device to close.".format(str(e)))
            self.__info.set_state(DeviceState.connected)
            raise e    
        self.__info.set_state(DeviceState.connected)

    @LogPerformances
    def stop_measurement(self):
        """Stops the measurement requested.

        :param file_id: id of the file to download.
        :type file_id: int
        :raises TMSiError: TMSiErrorCode.api_invalid_command if already sampling.
        :raises TMSiError: TMSiErrorCode.device_not_connected if not connected.
        """
        if self.__info.get_state() != DeviceState.sampling:
            raise TMSiError(error_code = TMSiErrorCode.api_invalid_command)
        TMSiLoggerActivity().log("TMSi-SDK->>{}: stop".format(self.__measurement.get_name()))
        try:
            self.__measurement.stop()
        except TMSiError as e:
            TMSiLogger().warning(message = "stop_measurement failed with error:\n{}\nDevice state moved to connected to allow the device to close.".format(str(e)))
            self.__info.set_state(DeviceState.connected)
            raise e    
        self.__info.set_state(DeviceState.connected)
    
    @LogPerformances
    def user_abort_download(self):
        """Interrupts the download after user abort command.
        """
        self.__user_abort = True
    
    @LogPerformances
    def __check_bandwidth(self):
        bw_requested = 0
        for channel in self.__config.get_active_channels():
            bw_requested += channel.get_channel_bandwidth()
        bw_available = self.__info.get_interface_bandwidth()
        if bw_available <= bw_requested:
            raise TMSiError(error_code = TMSiErrorCode.api_incompatible_configuration,
                            message = "The interface bandwidth is not compatible with the channel configuraton.")
    
    @LogPerformances
    def __download_impedance_report(self, file_id, filename):
        recording_metadata, impedances = self.__get_device_card_file_metadata(file_id)
        if recording_metadata.ImpAvailable == 0:
            return
        try:
            with open("{}.txt".format(filename), 'w') as f:
                f.write("Recorded Filename: {}\n\n".format(recording_metadata.RecFileName.decode()))
                f.write("Idx\t\tName\t\t\tImpedance\n")
                imp_channels = self.__config.get_channels()
                f.write("\n".join(["{}\t\t{}\t\t{} kOhm".format(i.ChanNr, imp_channels[i.ChanNr].get_channel_name(), i.Impedance) for i in impedances]))
        except Exception as e:
            raise TMSiError(TMSiErrorCode.file_writer_error)
    
    @LogPerformances
    def __initialize():
        from .saga_API import SagaSDK
        try:
            SagaDevice.__saga_sdk = SagaSDK
            for i in range (SagaDevice.__MAX_NUM_DEVICES):
                SagaDevice.__device_info_list.append(SagaInfo())
        except:
            SagaDevice.__saga_sdk = None
            raise TMSiError(
                error_code = TMSiErrorCode.api_no_driver)

    @LogPerformances
    def __get_device_card_file_list(self):
        from .saga_API import TMSiGetDeviceStorageList
        file_list = (2000 * TMSiDevRecList)()
        file_number = (c_uint)(0)
        self.__last_error_code = TMSiGetDeviceStorageList(
            self.__device_handle,
            pointer(file_list),
            len(file_list),
            pointer(file_number)
            )
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return_list = []
            for i in range(file_number.value):
                return_list.append(file_list[i])
            return return_list
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __get_device_card_file_metadata(self, file_id):
        from .saga_API import TMSiGetRecordingFile
        recording_metadata = TMSiDevRecDetails()
        impedance_report_list = (TMSiDevImpReport*self.__info.get_num_active_imp_channels())()
        impedance_report_list_len = self.__info.get_num_active_imp_channels()
        self.__last_error_code = TMSiGetRecordingFile(
            self.__device_handle,
            file_id,
            SampleControl.STARTSamplingDevice.value,
            pointer(recording_metadata),
            pointer(impedance_report_list),
            impedance_report_list_len)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            impedances = []
            for i in range(impedance_report_list_len):
                impedances.append(impedance_report_list[i])
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
        useless_1 = TMSiDevRecDetails()
        useless_2 = (TMSiDevImpReport*self.__info.get_num_active_imp_channels())()
        useless_2_len = self.__info.get_num_active_imp_channels()
        self.__last_error_code = TMSiGetRecordingFile(
            self.__device_handle,
            file_id,
            SampleControl.STOPSamplingDevice.value,
            pointer(useless_1),
            pointer(useless_2),
            useless_2_len)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return recording_metadata, impedances
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __get_device_card_recording_config(self):
        from .saga_API import TMSiGetDeviceAmbConfig
        config = TMSiDevRecCfg()
        self.__last_error_code = TMSiGetDeviceAmbConfig(
            self.__device_handle,
            pointer(config))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return config
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __get_device_configuration(self):
        from .saga_API import TMSiGetDeviceConfig
        n_channels = self.__info.get_num_channels()
        device_config = TMSiDevGetConfig()
        device_channel_list = (TMSiDevChDesc * n_channels)()
        self.__last_error_code = TMSiGetDeviceConfig(self.__device_handle, pointer(device_config), pointer(device_channel_list), n_channels)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            self.__info.set_device_config(device_config = device_config)
            self.__config.set_device_config(device_config = device_config)
            channels = []
            for dev_channel in device_channel_list:
                channel = SagaChannel()
                channel.set_channel_information(dev_channel)
                channels.append(channel)
            self.__config.set_channels(channels)
            self.__info.set_num_active_channels(len(self.__config.get_active_channels()))
            self.__info.set_num_active_imp_channels(len(self.__config.get_active_imp_channels()))
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
        self.__get_device_sensors()
        return device_config
    
    @LogPerformances
    def __get_full_device_status(self):
        from .saga_API import TMSiGetFullDeviceStatus
        dev_full_status_report = TMSiDevFullStatReport()
        dev_bat_report = (TMSiDevBatReport * _MAX_NUM_BATTERIES)()
        dev_time = TMSiTime()
        dev_storage_report = TMSiDevStorageReport()

        self.__last_error_code = TMSiGetFullDeviceStatus(self.__device_handle,
                                                         pointer(dev_full_status_report),
                                                         pointer(dev_bat_report),
                                                         _MAX_NUM_BATTERIES,
                                                         pointer(dev_time),
                                                         pointer(dev_storage_report))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            status = {}
            status["TotalSizeMB"] = dev_storage_report.TotalSizeMB
            status["UsedSizeMB"] = dev_storage_report.UsedSizeMB
            return status
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __get_device_sensors(self):
        from .saga_API import TMSiGetDeviceSensor
        device_sensor_list = (TMSiDevGetSens * self.__info.get_num_sensors())()
        sensor_list_len = c_ulong()
        self.__last_error_code = TMSiGetDeviceSensor(
            self.__device_handle, 
            pointer(device_sensor_list), 
            self.__info.get_num_sensors(),
            pointer(sensor_list_len))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            self.__update_sensor_list(device_sensor_list = device_sensor_list, sensor_list_len = sensor_list_len)
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __get_device_status(self):
        from .saga_API import TMSiGetDeviceStatus
        device_status_report = TMSiDevStatReport()
        self.__last_error_code = TMSiGetDeviceStatus(self.__device_handle, pointer(device_status_report))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            self.__info.set_device_status_report(device_status_report = device_status_report)
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __load_config_from_device(self):
        self.__get_device_status()
        self.__get_device_configuration()

    @LogPerformances
    def __reset_device_card(self):
        config = self.__get_device_card_recording_config()
        if config.StartControl != 4:
            tmp = config.StartControl
            config.StartControl = 4
            self.__set_device_card_recording_config(config = config, reset = True)
            config.StartControl = tmp
        self.__set_device_card_recording_config(config, reset = True)
        
    @LogPerformances
    def __set_device_card_recording_config(self, config, reset = False):
        from .saga_API import TMSiSetDeviceAmbConfig
        if config.StartControl != 0 and not reset:
            bw_button = 2_000_000
            bw_requested = 0
            for channel in self.__config.get_active_channels():
                bw_requested += channel.get_channel_bandwidth()
            if bw_requested > bw_button:
                raise TMSiError(error_code = TMSiErrorCode.api_incompatible_configuration,
                            message = "The interface bandwidth is not compatible with the channel configuraton.")
        self.__last_error_code = TMSiSetDeviceAmbConfig(
            self.__device_handle,
            pointer(config))
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
    
    @LogPerformances
    def __set_device_channel_names(self, names, indices):
        channels = self.__config.get_channels()
        for i in range(len(indices)):
            channels[indices[i]].set_channel_name(alternative_channel_name = names[i])

    @LogPerformances
    def __set_device_channel_sample_rates(self, channel_type, channel_divider):
        for channel in self.__config.get_channels():
            if channel.get_channel_divider() != -1:
                if ChannelType.all_types == channel_type or channel.get_channel_type() == channel_type:
                    channel.set_channel_divider(
                        divider = channel_divider,
                        base_sample_rate = self.__config.get_sample_rate())

    @LogPerformances
    def __set_device_config(
        self,
        base_sample_rate = None):
        from .saga_API import TMSiSetDeviceConfig
        dev_set_config = TMSiDevSetConfig()
        dev_set_config.DRSerialNumber = self.__info.get_dr_serial_number()
        dev_set_config.SetBaseSampleRateHz = self.__config.get_sample_rate() if base_sample_rate is None else base_sample_rate.value
        dev_set_config.NrOfChannels = self.__info.get_num_channels()
        dev_set_config.SetConfiguredInterface = self.__config.get_configured_interface()
        dev_set_config.SetTriggers = self.__config.get_triggers()
        dev_set_config.SetRefMethod = self.__config.get_reference_method().value
        dev_set_config.SetAutoRefMethod = self.__config.get_auto_reference_method().value
        dev_set_config.SetDRSyncOutDiv = self.__config.get_dr_sync_out_divider()
        dev_set_config.DRSyncOutDutyCycl = self.__config.get_dr_sync_out_duty_cycle()
        dev_set_config.SetRepairLogging = self.__config.get_repair_logging()
        dev_set_config.StoreAsDefault = 1 # Store always as default configuration
        dev_set_config.WebIfCtrl = 0

        default_pin = bytearray('0000', 'utf-8')
        dev_set_config.PinKey[:] = default_pin[:]

        dev_set_config.PerformFactoryReset = 0

        dev_channel_list = (TMSiDevSetChCfg * self.__info.get_num_channels())()
        for idx, saga_channel in enumerate(self.__config.get_channels()):

            dev_channel_list[idx].ChanNr = idx
            dev_channel_list[idx].ChanDivider = saga_channel.get_channel_divider()
            max_len = len( saga_channel.get_channel_name())
            name = bytearray(saga_channel.get_channel_name(), 'utf-8')
            dev_channel_list[idx].AltChanName[:max_len] = name[:max_len]

        self.__last_error_code = TMSiSetDeviceConfig(self.__device_handle, pointer(dev_set_config), pointer(dev_channel_list), self.__info.get_num_channels())
        if self.__last_error_code != TMSiDeviceRetVal.TMSI_OK:
            # Failure TMSiSetDeviceConfig()
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))
        
    @LogPerformances
    def __set_device_factory_default(self):
        from .saga_API import TMSiSetDeviceConfig
        dev_set_config = TMSiDevSetConfig()
        dev_set_config.DRSerialNumber = 0
        dev_set_config.NrOfChannels = 0
        dev_set_config.SetBaseSampleRateHz = 0
        dev_set_config.SetConfiguredInterface = 0
        dev_set_config.SetTriggers = 0
        dev_set_config.SetRefMethod = 0
        dev_set_config.SetAutoRefMethod = 0
        dev_set_config.SetDRSyncOutDiv = 0
        dev_set_config.DRSyncOutDutyCycl = 0
        dev_set_config.SetRepairLogging = 0
        dev_set_config.StoreAsDefault = 0
        dev_set_config.WebIfCtrl = 0
        default_pin = bytearray('0000', 'utf-8')
        dev_set_config.PinKey[:] = default_pin[:]
        dev_set_config.PerformFactoryReset = 1

        dev_set_channel = TMSiDevSetChCfg()
        dev_set_channel.ChanNr = 0
        dev_set_channel.ChanDivider = -1
        

        self.__last_error_code = TMSiSetDeviceConfig(self.__device_handle, pointer(dev_set_config), pointer(dev_set_channel), 1)
        if self.__last_error_code == TMSiDeviceRetVal.TMSI_OK:
            return
        else:
            raise TMSiError(
                error_code = TMSiErrorCode.device_error, 
                dll_error = DeviceErrorLookupTable(dll_response = self.__last_error_code))

    @LogPerformances
    def __update_sensor_list(self, device_sensor_list, sensor_list_len):
        sensor_list = []
        for i in range(sensor_list_len.value):
            sensor = SagaSensor()
            sensor.set_sensor_idx_total_channel_list(device_sensor_list[i].ChanNr)
            sensor.set_sensor_id(device_sensor_list[i].SensorID)
            sensor.set_sensor_IOMode(device_sensor_list[i].IOMode)
            idx = 0
            manufacturer_id, serial_nr, product_id, channel_count, additional_structs = \
                struct.unpack_from('<HIQBB', device_sensor_list[i].SensorMetaData, idx)
            if (channel_count > 0):
                if (self.__config.get_channels()[device_sensor_list[i].ChanNr].get_channel_type() == ChannelType.AUX):
                    # It concerns an AUX-channel-group: AUX-1, AUX-2 or AUX-3
                    sensor.set_sensor_manufacturer_id(manufacturer_id)
                    sensor.set_sensor_serial_nr(serial_nr)
                    sensor.set_sensor_product_id(product_id)
                    idx += 16
                    for j in range(channel_count):
                        struct_id = struct.unpack_from('<H', device_sensor_list[i].SensorMetaData, idx)
                        # Parse the data if it concerns a 'SensorDefaultChannel'
                        if (struct_id[0] == 0x0000):
                            idx += 2
                            chan_name, unit_name, exp, gain, offset = struct.unpack_from('<10s10shff', device_sensor_list[i].SensorMetaData, idx)

                            sensor.set_sensor_name(chan_name)
                            sensor.set_sensor_unit_name(unit_name)
                            sensor.set_sensor_exp(exp)
                            sensor.set_sensor_gain(gain)
                            sensor.set_sensor_offset(offset)

                            # append sensor-into to the device-sensor-list and ...
                            sensor_list.append(copy(sensor))
                            # attach a copy of the sensor-object also to the specified channel
                            self.__config.get_channels()[sensor.get_sensor_idx_total_channel_list()].set_sensor_information(sensor = copy(sensor))

                            # Prepare for next sensor-channel
                            sensor.set_sensor_idx_total_channel_list(sensor.get_sensor_idx_total_channel_list() + 1)
                            idx += 30
                        else:
                            # ran into an 'empty struct', no more sensor info expected
                            break
            else:
                # Always add the sensor-data to the device-sensor-list
                sensor_list.append(copy(sensor))
                # Add sensor-object to both BIP-channels when a sensor is detected on a BIP-channel
                if (self.__config.get_channels()[device_sensor_list[i].ChanNr].get_channel_type() == ChannelType.BIP) and (sensor.get_sensor_id() != -1):
                    self.__config.get_channels()[device_sensor_list[i].ChanNr].set_sensor_information(sensor = copy(sensor), bipolar = True)
                    sensor.set_sensor_idx_total_channel_list(sensor.get_sensor_idx_total_channel_list() + 1)
                    sensor_list.append(copy(sensor))
                    self.__config.get_channels()[sensor.get_sensor_idx_total_channel_list()].set_sensor_information(sensor = copy(sensor), bipolar = True)
        self.__config.set_sensors(sensors = sensor)

    