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
 * @file saga_API_lookup_table.py
 * @brief 
 * Lookup table to translate codes to messages.
 */


'''
def DeviceErrorLookupTable(dll_response):
    """Translate code to string

    :param dll_response: dll_response to translate
    :type dll_response: TMSiRetVal
    :return: dll_response translated to string
    :rtype: string
    """
    _lookup_table = {
        '0x00000000':"All Ok positive ACK.",

        '0x01010001':"DR reported 'Checksum error in received block'.",
        '0x01010002':"DR reported 'Unknown command'.",
        '0x01010003':"DR reported 'Response timeout'.",
        '0x01010004':"DR reported 'Device busy try again in x msec'.",
        '0x01010005':"DR reported 'Command not supported over current interface'.",
        '0x01010006':"DR reported 'Command not possible, device is recording'.",
        '0x01010007':"DR reported 'Device not available'.",
        '0x01010008':"DR reported 'Configuration error'.",
        '0x01030001':"DR reported 'Command not supported by Channel'.",
        '0x01030002':"DR reported 'Illegal start control for ambulant recording'.",

        '0x02010001':"DS reported 'Checksum error in received block'.",
        '0x02010002':"DS reported 'Unknown command'.",
        '0x02010003':"DS reported 'Response timeout'.",
        '0x02010004':"DS reported 'Device busy try again in x msec'.",
        '0x02010005':"DS reported 'Command not supported over current interface'.",
        '0x02010006':"DS reported 'Unexpected response'.",
        '0x02010007':"DS reported 'Device not available'.",
        '0x02010008':"DS reported 'Interface not available'.",
        '0x02010009':"DS reported 'Command not allowed in current mode'.",
        '0x0201000A':"DS reported 'Processing error'.",
        '0x0201000B':"DS reported 'Unknown internal error'.",
        '0x0201000C':"DS reports that data request does not fit with one Device Api Packet.",
        '0x0201000D':"DS reports that DR is already opened.",
        '0x0201000E':"DS reports that it’s API-access is locked.",

        '0x03001000':"DLL Function is declared, but not yet implemented.",
        '0x03001001':"DLL Function called with invalid parameters.",
        '0x03001002':"Incorrect checksum of response message.",
        '0x03001003':"DLL Function failed because of header failure.",
        '0x03001004':"DLL Function failed because an underlying process failed.",
        '0x03001005':"DLL Function falled with a too small buffer.",
        '0x03001006':"DLL Function falled with a Handle that's not assigned to a device.",
        '0x03002000':"DLL Function failed because could not open selected interface.",
        '0x03002001':"DLL Function failed because could not close selected interface.",
        '0x03002002':"DLL Function failed because could not send command-data.",
        '0x03002003':"DLL Function failed because could not receive data.",
        '0x03002004':"DLL Function failed because commination timed out.",
        '0x03002005':"Lost connection to DS, USB / Ethernet disconnect.",
        '0x03002006':"DLL reports that it’s API-access is locked.",
        }    
    
    try:
        hex_representation = '0x{:08X}'.format(dll_response.value)
        return "Error code {}: {}".format(hex_representation, _lookup_table[hex_representation])
    except:
        return "Unknown Error Code. Code {} not listed.".format(hex_representation) 
        

