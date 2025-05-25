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
 * @file apex_config.py 
 * @brief 
 * APEX Configuration object.
 */


'''

import xml.etree.ElementTree as ET
from xml.dom import minidom

from .....tmsi_utilities.tmsi_logger import TMSiLogger
from ..apex_API_enums import *

class ApexConfig():
    """Class to handle the configuration of the Apex."""
    def __init__(self):
        """Initialize the configuration.
        """
        self.__base_sample_rate = ApexBaseSampleRate.Decimal
        self.__channels = []
        self.__impedance_channels = []
        self.__impedance_limit = 0
        self.__live_impedance = ApexLiveImpedance.Off
        self.__sampling_frequency = 0

    def export_to_xml(self, filename):
        """Export the current configuration to xml file.

        :param filename: filename where to save the configuration.
        :type filename: str
        :return: True if succeded, False if failed.
        :rtype: bool
        """
        try:
            root = ET.Element("ApexConfig")
            xml_device = ET.SubElement(root, "Device")
            ET.SubElement(xml_device, "BaseSampleRate").text = str(self.__base_sample_rate)
            ET.SubElement(xml_device, "ImpedanceLimit").text = str(self.__impedance_limit)
            ET.SubElement(xml_device, "LiveImpedance").text = str(self.__live_impedance)
            xml_channels = ET.SubElement(root, "Channels")
            for idx, channel in enumerate(self.__channels):
                xml_channel = ET.SubElement(xml_channels, "Channel")
                ET.SubElement(xml_channel, "ChanIdx").text = str(idx)
                ET.SubElement(xml_channel, "AltChanName").text = channel.get_channel_name()
                ET.SubElement(xml_channel, "ReferenceStatus").text = str(channel.is_reference()) 
            xml_data = ApexConfig.__prettify(root)
            xml_file = open(filename, "w")
            xml_file.write(xml_data)
            return True
        except:
            return False

    def import_from_xml(self, filename):
        """Import the configuration from a file to the device.

        :param filename: filename where to take the configuration from.
        :type filename: str
        :return: True if succeded, False if failed, error message.
        :rtype: bool
        """
        error_message = "Error during xml read."
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            
            # check device type
            if root.tag != "ApexConfig":
                error_message = "IMPOSSIBLE TO LOAD FILE! It is not a APEX configuration file."
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
                        if subelem.tag == "BaseSampleRate":
                            self.__base_sample_rate = int(subelem.text)
                        if subelem.tag == "ImpedanceLimit":
                            self.__impedance_limit = int(subelem.text)
                        if subelem.tag == "LiveImpedance":
                            self.__live_impedance = int(subelem.text)
                    if elem.tag == "Channels":
                        if subelem.tag == "Channel":
                            found = False
                            idx = subelem.find("ChanIdx")
                            if idx is None:
                                continue
                            idx = int(idx.text)
                            self.__channels[idx].set_channel_name(
                                alternative_channel_name = subelem.find("AltChanName").text
                            )
                            reference = subelem.find("ReferenceStatus").text
                            if reference != 'None':
                                self.__channels[idx].set_reference(int(reference))
                            else:
                                self.__channels[idx].set_reference(0)
            return True, None
        except Exception as exec:
            return False, error_message
    
    def get_channels(self):
        """Get channels of the device.

        :return: list of apex channels.
        :rtype: list[ApexChannel]
        """
        return self.__channels

    def get_impedance_channels(self):
        """Get impedance channels of the device.

        :return: list of impedance channels.
        :rtype: list[ApexImpedanceChannel]
        """
        return self.__impedance_channels
    
    def get_impedance_limit(self):
        """Get the impedance limit.

        :return: impedance limit of the device
        :rtype: int
        """
        return self.__impedance_limit

    def get_live_impedance(self):
        """Get live impedance

        :return: live impedance on or off
        :rtype: ApexLiveImpedance
        """
        return self.__live_impedance

    def get_sample_rate(self):
        """Get sample rate

        :return: base sample rate.
        :rtype: ApexBaseSampleRate
        """
        return self.__base_sample_rate

    def get_sampling_frequency(self):
        """Get sampling frequency

        :return: sampling frequency
        :rtype: int
        """
        return self.__sampling_frequency
    
    def set_channels(self, channels):
        """Set channels of the device

        :param channels: list of apex channels
        :type channels: list[ApexChannels]
        """
        self.__channels = channels

    def set_device_impedance_channels(self, channels):
        """Set impedance channels of the device.

        :param channels: list of impedance channels.
        :type channels: list[ApexImpedanceChannel]
        """
        self.__impedance_channels = channels
        
    def set_device_sampling_config(self, device_sampling_config):
        """Set device sampling configuration

        :param device_sampling_config: sampling configuration for the device.
        :type device_sampling_config: TMSiDevSamplingCfg
        """
        self.__base_sample_rate = device_sampling_config.BaseSampleRate
        self.__impedance_limit = device_sampling_config.ImpedanceLimit
        self.__live_impedance = device_sampling_config.LiveImpedance

    def set_sampling_frequency(self, sampling_frequency):
        """Set the sampling frequency

        :param sampling_frequency: sampling frequency
        :type sampling_frequency: int
        """
        self.__sampling_frequency = sampling_frequency

    def __prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

