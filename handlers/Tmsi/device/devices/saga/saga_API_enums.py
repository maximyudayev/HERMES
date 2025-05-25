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
 * @file saga_API_enums.py 
 * @brief 
 * Enumerable classes useful for the use of the API.
 */


'''

from enum import Enum, unique
#---------------------------------------------------------------------

# Error codes

#---
# 0x01xxxxxx = FE API related, 0x02xxxxxx is reserved for USER API
# Error codes are categorized as:
# 0x0101xxxx # Gen. System status
# 0x0102xxxx # Hardware related status
# 0x0103xxxx # Firmware related status
#---

#---
# Defined status codes are:
# Generic Device status codes for the DR 0x0101xxxx
# Generic Device status codes for the DS 0x0201xxxx
#
# Hardware specific status codes for the DR 0x0102xxxx
# Hardware specific status codes for the DS 0x0202xxxx
#
# Firmware specific status codes for the DR 0x0103xxxx
# Firmware specific status codes for the DS 0x0203xxxx
#
#---
# Each DLL API function on the TMSi Device API has a return value TMSiDeviceRetVal.
@unique
class TMSiDeviceRetVal(Enum):
	TMSI_OK = 0x00000000					# All Ok positive ACK.
	
	TMSI_DR_CHECKSUM_ERROR = 0x01010001						# DR reported "Checksum error in received block".
	TMSI_DR_UNKNOWN_COMMAND = 0x01010002 					# DR reported "Unknown command".
	TMSI_DR_RESPONSE_TIMEMOUT = 0x01010003 					# DR reported "Response timeout".
	TMSI_DR_DEVICE_BUSY = 0x01010004 						# DR reported "Device busy try again in x msec".
	TMSI_DR_COMMAND_NOT_SUPPORTED = 0x01010005 				# DR reported "Command not supported over current interface".
	TMSI_DR_COMMAND_NOT_POSSIBLE = 0x01010006 				# DR reported "Command not possible, device is recording".
	TMSI_DR_DEVICE_NOT_AVAILABLE = 0x01010007 				# DR reported "Device not available".
	TMSI_DR_CONFIGURATION_ERROR = 0x01010008 				# DR reported 'Configuration error".
	TMSI_DR_COMMAND_NOT_SUPPORTED_BY_CHANNEL = 0x01030001 	# DR reported "Command not supported by Channel".
	TMSI_DR_AMBREC_ILLEGAL_START_CTRL = 0x01030002 			# DR reported "Illegal start control for ambulant recording".
	
	TMSI_DS_CHECKSUM_ERROR = 0x02010001						# DS reported "Checksum error in received block".
	TMSI_DS_UNKNOWN_COMMAND = 0x02010002 					# DS reported "Unknown command".
	TMSI_DS_RESPONSE_TIMEMOUT = 0x02010003 					# DS reported "Response timeout".
	TMSI_DS_DEVICE_BUSY = 0x02010004 						# DS reported "Device busy try again in x msec".
	TMSI_DS_COMMAND_NOT_SUPPORTED = 0x02010005 				# DS reported "Command not supported over current interface".
	TMSI_DS_UNEXPECTED_RESPONSE = 0x02010006 				# DS reported "Unexpected response".
	TMSI_DS_DEVICE_NOT_AVAILABLE = 0x02010007 				# DS reported "Device not available".
	TMSI_DS_INTERFACE_NOT_AVAILABLE = 0x02010008 			# DS reported "Interface not available".
	TMSI_DS_COMMAND_NOT_ALLOWED = 0x02010009 				# DS reported "Command not allowed in current mode".
	TMSI_DS_PROCESSING_ERROR = 0x0201000A 					# DS reported "Processing error".
	TMSI_DS_UNKNOWN_INTERNAL_ERROR = 0x0201000B 			# DS reported "Unknown internal error".
	TMSI_DS_PACKET_LENGTH_ERROR = 0x0201000C 				# DS reports that data request does not fit with one Device Api Packet
	TMSI_DS_DEVICE_ALREADY_OPEN = 0x0201000D 				# DS reports that DR is already opened.
	


	# Additional defines below for DLL error types.
	TMSI_DLL_NOT_IMPLEMENTED = 0x03001000 					# DLL Function is declared, but not yet implemented
	TMSI_DLL_INVALID_PARAM = 0x03001001 					# DLL Function called with invalid parameters
	TMSI_DLL_CHECKSUM_ERROR = 0x03001002
	TMSI_DLL_ETH_HEADER_ERROR = 0x03001003
	TMSI_DLL_INTERNAL_ERROR = 0x03001004 					# DLL Function failed because an underlying process failed
	TMSI_DLL_BUFFER_ERROR = 0x03001005 						# DLL Function called with a too small buffer
	TMSI_DLL_INVALID_HANDLE = 0x03001006 					# DLL Function called with a Handle that's not assigned to a device
	TMSI_DLL_INTF_OPEN_ERROR = 0x03002000
	TMSI_DLL_INTF_CLOSE_ERROR = 0x03002001
	TMSI_DLL_INTF_SEND_ERROR = 0x03002002
	TMSI_DLL_INTF_RECV_ERROR = 0x03002003
	TMSI_DLL_INTF_RECV_TIMEOUT = 0x03002004
	TMSI_DLL_LOST_CONNECTION = 0x03002005 					# Lost connection to DS, USB / Ethernet disconnect.



# Communication interface used
# 0 = Unknown, 1=USB 2=Nework, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
@unique
class TMSiInterface(Enum):
	IF_TYPE_UNKNOWN = 0
	IF_TYPE_USB = 1
	IF_TYPE_NETWORK = 2
	IF_TYPE_WIFI = 3
	IF_TYPE_ELECTRICAL = 4
	IF_TYPE_OPTICAL = 5
	IF_TYPE_BLUETOOTH = 6


#---
# SampleControl, enum for SetSamplingMode values.
#---
@unique
class SampleControl(Enum):
    STOPSamplingDevice = 0
    STARTSamplingDevice = 1
    STOPWiFiStream = 2


#---
# ImpedanceControl, enum for SetImpedanceMode values.
#---
@unique
class ImpedanceControl(Enum):
    ImpedanceStop = 0
    ImpedanceStart = 1

#---
# FWStatus, enum for FWStatus values.
#---
@unique
class FirmwareStatus(Enum):
    FWStatus_Unknown = -1
    All_OK = 0
    Upgrading = 1
    Verify_OK = 2
    Verify_Fail = 3


#---
# FWAction enum used to specify the action to perform during FWupdate.
#---
@unique
class FWAction(Enum):
    FWAct_Unknown = 0
    FWAct_Flash_Reboot = 1
    FWAct_ABORT = 2

@unique
class RefMethod(Enum):
     Common = 0
     Average = 1
     
@unique
class AutoRefMethod(Enum):
     Fixed = 0
     Average = 1
     
@unique
class SagaBaseSampleRate(Enum):
    Decimal = 4000
    Binary = 4096

@unique
class SagaStartCardRecording(Enum):
    Off = 0
    Time = 1
    Button = 8
    Remote = 16
    
@unique
class SagaStringLengths(Enum):
    AltChanName = 10
    PrefixFileName = 16
    UserString = 64
    PatientString = 128

