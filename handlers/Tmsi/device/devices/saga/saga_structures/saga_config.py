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
 * @file saga_config.py 
 * @brief 
 * SAGA Configuration object.
 */


'''

import xml.etree.ElementTree as ET
from xml.dom import minidom

from .....tmsi_utilities.tmsi_logger import TMSiLogger
from ..saga_API_enums import *

class SagaConfig():
    """Class to handle the configuration of the Saga."""
    def __init__(self):
        """Initialize the configuration.
        """
        self.__base_sample_rate = 4000
        self.__sampling_frequency = 0
        self.__channels = []
        self.__active_channels = []
        self.__active_imp_channels = []
        self.__sensors = []
        self.__triggers = 0
        self.__reference_method = RefMethod.Common
        self.__auto_reference_method = AutoRefMethod.Fixed
        self.__device_name = "-"
        self.__configured_interface = None
        self.__dr_sync_out_divider = -1
        self.__dr_sync_out_duty_cycle = None
        self.__masked_channels = []
        self.__mask_functions = []

    def export_to_xml(self, filename):
        """Export the current configuration to xml file.

        :param filename: filename where to save the configuration.
        :type filename: str
        :return: True if succeded, False if failed.
        :rtype: bool
        """
        try:
            root = ET.Element("SagaConfig")
            xml_device = ET.SubElement(root, "Device")
            ET.SubElement(xml_device, "BaseSampleRateHz").text = str(self.__base_sample_rate)
            ET.SubElement(xml_device, "ConfiguredInterface").text = str(self.__configured_interface)
            ET.SubElement(xml_device, "Triggers").text = str(self.__triggers)
            ET.SubElement(xml_device, "ReferenceMethod").text = str(self.__reference_method.value)
            ET.SubElement(xml_device, "AutoReferenceMethod").text = str(self.__auto_reference_method.value)
            ET.SubElement(xml_device, "DRSyncOutDiv").text = str(self.__dr_sync_out_divider)
            ET.SubElement(xml_device, "DRSyncOutDutyCycl").text = str(self.__dr_sync_out_duty_cycle)
            ET.SubElement(xml_device, "RepairLogging").text = str(self.__repair_logging)

            xml_channels = ET.SubElement(root, "Channels")
            for idx, channel in enumerate(self.__channels):
                xml_channel = ET.SubElement(xml_channels, "Channel")
                ET.SubElement(xml_channel, "ChanNr").text = str(idx)
                ET.SubElement(xml_channel, "AltChanName").text = channel.get_channel_name()
                ET.SubElement(xml_channel, "ChanDivider").text = str(channel.get_channel_divider())
            xml_data = SagaConfig.__prettify(root)
            xml_file = open(filename, "w")
            xml_file.write(xml_data)
            return True
        except:
            return False

    def import_from_xml(self, filename):
        """Import the configuration from a file to the device.

        :param filename: filename where to take the configuration from.
        :type filename: str
        :return: True if succeded, False if failed, error_message.
        :rtype: bool
        """
        error_message = "Error during xml read."
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            
            # check device type
            if root.tag != "SagaConfig" and root.tag != "DeviceConfig":
                error_message = "IMPOSSIBLE TO LOAD FILE! It is not a SAGA configuration file."
                TMSiLogger().warning(error_message)
                return False, error_message
            
            # check if imported file has the correct number of channels
            channel_counter = 0
            for elem in root:
                n_channels = len(self.__channels)
                for subelem in elem:
                    if elem.tag == "Channels" and subelem.tag == "Channel":
                        channel_counter += 1
            if channel_counter == 0:
                # all good, configuring only the device
                pass
            elif channel_counter < n_channels:
                error_message = "Configuration file loaded does not have the full list of channels."
                TMSiLogger().warning(error_message)
                return False, error_message
            elif channel_counter > n_channels:
                error_message = "Configuration file loaded is not compatible with the device in use, too many channels set."
                TMSiLogger().warning(error_message)
                return False, error_message
            
            # fill configuration structure
            for elem in root:
                for subelem in elem:
                    if elem.tag == "Device":
                        if subelem.tag == "BaseSampleRateHz":
                            self.__base_sample_rate = int(subelem.text)
                        if subelem.tag == "ConfiguredInterface":
                            self.__configured_interface = int(subelem.text)
                        if subelem.tag == "Triggers":
                            self.__triggers = int(subelem.text)
                        if subelem.tag == "ReferenceMethod":
                            self.__reference_method = RefMethod(int(subelem.text))
                        if subelem.tag == "AutoReferenceMethod":
                            self.__auto_reference_method = AutoRefMethod(int(subelem.text))
                        if subelem.tag == "DRSyncOutDiv":
                            self.__dr_sync_out_divider = int(subelem.text)
                        if subelem.tag == "DRSyncOutDutyCycl":
                            self.__dr_sync_out_duty_cycle = int(subelem.text)
                        if subelem.tag == "RepairLogging":
                            self.__repair_logging = int(subelem.text)
                    if elem.tag == "Channels":
                        if subelem.tag == "Channel":
                            found = False
                            idx = subelem.find("ChanNr")
                            if idx is None:
                                continue
                            idx = int(idx.text)
                            self.__channels[idx].set_channel_name(
                                alternative_channel_name = subelem.find("AltChanName").text
                            )
                            self.__channels[idx].set_channel_divider(
                                divider = int(subelem.find("ChanDivider").text),
                                base_sample_rate = self.__base_sample_rate
                            )
            return True, None
        except Exception as e:
            TMSiLogger().warning("{}".format(e))
            return False, error_message
    
    def get_active_channels(self):
        """Get active channels of the device.

        :return: list of saga active channels.
        :rtype: list[SagaChannel]
        """
        return self.__active_channels

    def get_active_imp_channels(self):
        """Get active impedance channels of the device.

        :return: list of saga active impedance channels.
        :rtype: list[SagaChannel]
        """
        return self.__active_imp_channels

    def get_auto_reference_method(self):
        """Get auto reference method

        :return: autoreference method
        :rtype: AutoRefMethog
        """
        return self.__auto_reference_method

    def get_channels(self):
        """Get channels of the device.

        :return: list of saga channels.
        :rtype: list[SagaChannel]
        """
        return self.__channels

    def get_configured_interface(self):
        """Get configured interface of the device.

        :return: configured interface
        :rtype: int
        """
        return self.__configured_interface

    def get_dr_sync_out_divider(self):
        """Get dr_sync_out_divider

        :return: dr_sync_out_divider
        :rtype: int
        """
        return self.__dr_sync_out_divider

    def get_dr_sync_out_duty_cycle(self):
        """Get dr_sync_out_duty_cycle

        :return: dr_sync_out_duty_cycle
        :rtype: int
        """
        return self.__dr_sync_out_duty_cycle

    def get_impedance_channels(self):
        """Get impedance channels of the device.

        :return: list of impedance channels.
        :rtype: list[SagaImpedanceChannel]
        """
        return self.__impedance_channels
    
    def get_impedance_limit(self):
        """Get the impedance limit.

        :return: impedance limit of the device
        :rtype: int
        """
        return self.__impedance_limit

    def get_mask_info(self):
        """Get mask info

        :return: dictionary with channels and masks
        :rtype: dict
        """
        return {"channels": self.__masked_channels, "functions": self.__mask_functions}
    
    def get_reference_method(self):
        """Get reference method

        :return: reference method
        :rtype: RefMethog
        """
        return self.__reference_method
    
    def get_repair_logging(self):
        """Get repair logging

        :return: repair logging
        :rtype: int
        """
        return self.__repair_logging

    def get_sample_rate(self):
        """Get sample rate

        :return: base sample rate.
        :rtype: int
        """
        return self.__base_sample_rate

    def get_sampling_frequency(self):
        """Get sampling frequency

        :return: sampling frequency
        :rtype: int
        """
        return self.__sampling_frequency
    
    def get_triggers(self):
        """Get triggers

        :return: 1 if triggers are enabled, 0 otherwise
        :rtype: int
        """
        return self.__triggers
    
    def set_channels(self, channels):
        """Set channels of the device

        :param channels: list of saga channels
        :type channels: list[SagaChannels]
        """
        self.__channels = channels
        self.__sampling_frequency = int(self.__base_sample_rate / (2**self.__channels[-1].get_channel_divider()))
        self.__active_channels = [ch for ch in channels if ch.get_channel_divider() != -1]
        self.__active_imp_channels = [ch for ch in channels if ch.get_channel_imp_divider() != -1]
        for i in range(len(self.__active_imp_channels)):
            self.__active_imp_channels[i].set_channel_index(index = i)

    def set_sensors(self, sensors):
        """Set sensors of the device

        :param sensors: list of sensors
        :type sensors: list[SagaSensors]
        """
        self.__sensors = sensors
    
    def set_configured_interface(self, configured_interface):
        self.__configured_interface = configured_interface.value

    def set_device_auto_reference_method(self, auto_reference_method):
        """Set auto reference method of the device.

        :param auto_reference_method: auto_reference_method.
        :type auto_reference_method: AutoReferenceMethod
        """
        self.__auto_reference_method = auto_reference_method
        
    def set_device_config(self, device_config):
        """Set device configuration

        :param device_config: configuration of the device.
        :type device_config: TMSiDevGetConfig
        """
        self.__base_sample_rate = device_config.BaseSampleRateHz
        self.__triggers = device_config.TriggersEnabled
        self.__reference_method = RefMethod(device_config.RefMethod)
        self.__auto_reference_method = AutoRefMethod(device_config.AutoRefMethod)
        self.__device_name = device_config.DeviceName.decode('windows-1252')
        self.__dr_sync_out_divider = device_config.DRSyncOutDiv
        self.__dr_sync_out_duty_cycle = device_config.DRSyncOutDutyCycl
        self.__repair_logging = device_config.RepairLogging
        self.__configured_interface = device_config.ConfiguredInterface

    def set_device_impedance_channels(self, channels):
        """Set impedance channels of the device.

        :param channels: list of impedance channels.
        :type channels: list[SagaImpedanceChannel]
        """
        self.__impedance_channels = channels
        
    def set_device_reference_method(self, reference_method):
        """Set reference method of the device.

        :param reference_method: reference_method.
        :type reference_method: ReferenceMethod
        """
        self.__reference_method = reference_method
        
    def set_dr_sync_out_divider(self, divider):
        """Set dr sync out divider

        :param divider: divider of the sync out
        :type divider: int
        """
        self.__dr_sync_out_divider = divider

    def set_dr_sync_out_duty_cycle(self, duty_cycle):
        """Set dr sync out duty cycle

        :param duty_cycle: duty cycle of the sync out
        :type duty_cycle: int
        """
        self.__dr_sync_out_duty_cycle = duty_cycle
    
    def set_mask_info(self, channels, functions):
        """Set mask info

        :param channels: list of indices of channels
        :type channels: list[int]
        :param functions: list of functions to apply
        :type functions: list[function]
        """
        self.__mask_functions = functions
        self.__masked_channels = channels
    
    def set_repair_logging(self, enable_repair_logging = True):
        """Set the repair logging

        :param enable_repair_logging: enable or not the repair logging, defaults to True
        :type enable_repair_logging: bool, optional
        """
        if enable_repair_logging:
            self.__repair_logging = 1
        else:
            self.__repair_logging = 0
    
    def set_triggers(self, triggers):
        """Set triggers

        :param triggers: True if enabled, False otherwise 
        :type triggers: bool
        """
        if triggers:
            self.__triggers = 1
        else:
            self.__triggers = 0
    
    def __prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

