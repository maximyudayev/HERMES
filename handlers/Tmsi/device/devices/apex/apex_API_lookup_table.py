'''
(c) 2024 Twente Medical Systems International B.V., Oldenzaal The Netherlands

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
 * @file apex_API_lookup_table.py
 * @brief 
 * Lookup table to translate codes to messages.
 */


'''
def DeviceErrorLookupTable(dll_response):
    """Translate code to string

    :param code: code to translate
    :type code: string
    :return: code translated to string
    :rtype: string
    """
    _lookup_table = {'0x00000000': 'OK.',
        '0x01010001': "DR reported 'ChecksumError'.",
        '0x01010002': "DR reported 'UnknownCommand'.",
        '0x01010003': "DR reported 'ResponseTimeout'.",
        '0x01010004': "DR reported 'InterfaceAlreadyOpen'.",
        '0x01010005': "DR reported 'UnknownCommandForInterface'.",
        '0x01010006': "DR reported 'DeviceRecording'.",
        '0x01010007': "DR reported 'ConfigError'.",
        '0x01010008': "DR reported 'DeviceLocked'.",
        '0x01010009': "DR reported 'ServiceModeLocked'.",
        
        '0x03001001': "DLL function failed with message 'DeviceNotAvailable'.",
        '0x03001002': "DLL function failed with message 'InterfaceAlreadyOpen'.",
        '0x03001003': "DLL function failed with message 'DeviceNotPaired'.",
        '0x03001004': "DLL function failed with message 'DeviceAlreadyPaired'.",
        '0x03001005': "DLL function failed with message 'NotImplemented'.",
        '0x03001006': "DLL function failed with message 'InvalidParameter'.",
        '0x03001007': "DLL function failed with message 'ChecksumError'.",
        '0x03001008': "DLL function failed with message 'InternalError'.",
        '0x03001009': "DLL function failed with message 'BufferError'.",
        '0x0300100A': "DLL function failed with message 'InvalidHandle'.",
        '0x0300100B': "DLL function failed with message 'InterfaceOpenError'.",
        '0x0300100C': "DLL function failed with message 'InterfaceCloseError'.",
        '0x0300100E': "DLL function failed with message 'InterfaceSendError'.",
        '0x0300100F': "DLL function failed with message 'InterfaceReceiveError'.",
        '0x03001010': "DLL function failed with message 'InterfaceTimeout'.",
        '0x03001011': "DLL function failed with message 'CommandInProgress'.",
        '0x03001012': "DLL function failed with message 'NoEventAvailable'.",
        '0x03001013': "DLL function failed with message 'InvalidCardFileID'.",
        '0x03001014': "DLL function failed with message 'CanNotDecodeSampleData'.",
        } 

    try:
        hex_representation = '0x{:08X}'.format(dll_response.value)
        return "Error code {}: {}".format(hex_representation, _lookup_table[hex_representation])
    except:
        return "Unknown Error Code. Code {} not listed.".format(hex_representation) 