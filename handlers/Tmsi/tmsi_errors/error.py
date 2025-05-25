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
 * @file error.py 
 * @brief 
 * TMSi Error interface.
 */


'''

from enum import Enum, unique

@unique
class TMSiErrorCode(Enum):
    """TMSi error codes available"""
    general_error = 0
    missing_dll = 1
    already_in_use_dll = 2
    device_error = 100
    device_not_connected = 101
    no_devices_found = 102
    api_no_driver = 200
    api_incorrect_argument = 201
    api_invalid_command = 202
    api_incompatible_configuration = 203
    file_writer_error = 300
    file_import_error = 301

class TMSiError(Exception):
    """Class to handle TMSi errors."""
    def __init__(self, error_code, dll_error = None, message = ""):
        """Initialize TMSi error.

        :param error_code: error code
        :type error_code: TMSiErrorCode
        :param dll_error_code: error code coming from the dll, defaults to None
        :type dll_error_code: TMSiDeviceRetVal, optional
        """
        self.dll_error =None
        self.error_code = error_code
        self.message = message
        self.dll_error = dll_error
        
    def __str__(self):
        if self.dll_error:
            return "TMSiSDK Error code {}: {} \nDLL {}\n{}".format(
                self.error_code.value,
                SdkErrorLookupTable(str(self.error_code.value)),
                self.dll_error,
                self.message)
        return "TMSiSDK Error code {}: {}\n{}".format(
                self.error_code.value,
                SdkErrorLookupTable(str(self.error_code.value)),
                self.message)

def SdkErrorLookupTable(code):
    """Translate code to string

    :param code: code to translate
    :type code: string
    :return: code translated to string
    :rtype: string
    """
    _lookup_table = {
        "0" : "general error",
        "1" : "missing dll",
        "2" : "already in use dll",
        "100" : "device error",
        "101" : "device not connected",
        "102" : "no devices found",
        "200" : "api no driver",
        "201" : "api incorrect argument",
        "202" : "api invalid command",
        "203" : "api incompatible configuration",
        "300" : "file writer error",
        "301" : "file import error"}
    
    try:
        return _lookup_table[code]
    except:
        return "Unknown Error Code. Code {} not listed.".format(code)