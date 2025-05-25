'''
(c) 2023 Twente Medical Systems International B.V., Oldenzaal The Netherlands

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
 * @file saga_API_structures.py 
 * @brief 
 * Structures useful for the use of the API.
 */


'''

from ctypes import *
# Protocol block definitions

#---
# TMSiDevList
#---
class TMSiDevList(Structure):
    _pack_=2
    _fields_ = [
        ("TMSiDeviceID", c_ushort),     # Unique ID to identify device, used to open device.
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRAvailable", c_ushort),      # defined as 0 = DR_Offline, 1 = DR_Online
        ("DRSerialNr", c_uint),         # The DR serial number.
    ]


#---
# TMSiDevStatReport
#---
class TMSiDevStatReport(Structure):
    _pack_=2
    _fields_ = [
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRSerialNr", c_uint),         # The DR serial number.
        ("DSInterface", c_ushort),      # Communication interface on DS used, 0 = Unknown, 1=USB, 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("DRInterface", c_ushort),      # Communication interface on DR used, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("DSDevAPIVersion", c_ushort),  # Current Device-API version used by DS, V00.00
        ("DRAvailable", c_ushort),      # defined as 0 = FE_Offline, 1 = FE_Online
        ("NrOfBatteries", c_ushort),    # Nr of batteries, indicates nr of TMSiBatReport.
        ("NrOfChannels", c_ushort),     # Total number of channels for the device.
    ]


#---
# TMSiDevFullStatReport
#---
class TMSiDevFullStatReport(Structure):
    _pack_=2
    _fields_ = [
        ("DSSerialNr", c_uint),         # The DS serial number.
        ("DRSerialNr", c_uint),         # The DR serial number.
        ("DRWiFiPaired", c_uint),       # The serial number of paired DR over WiFi.
        ("KeepAliveTimeout", c_short),  # DR Idle powerdown in sec. -1 disabled.
        ("PowerState", c_ushort),       # 0 = Unknown, 1 = External, 2 = Battery.
        ("DSTemp", c_short),            # DS temperature in degree Celsius.
        ("DRTemp", c_short),            # DR temperature in degree Celsius.
    ]


#---
# TMSiDevBatReport
#---
class TMSiDevBatReport(Structure):
    _pack_=2
    _fields_ = [
        ("BatID", c_short),                 # 0 unknown, 1, 2 etc.
        ("BatTemp", c_short),               # Battery temperature.
        ("BatVoltage", c_short),            # Battery voltage in mV.
        ("BatRemainingCapacity", c_short),  # Available battery capacity in mAh.
        ("BatFullChargeCapacity", c_short), # Max battery capacity in mAh.
        ("BatAverageCurrent", c_short),     # Current going in or out of the battery in mA.
        ("BatTimeToEmpty", c_short),        # Estimated remaining minutes before empty in min.
        ("BatStateOfCharge", c_short),      # Estimated capacity in %.
        ("BatStateOfHealth", c_short),      # Estimated battery health in %.
        ("BatCycleCount", c_short),         # Battery charge cycles.
    ]


#---
# TMSiTime
#---
class TMSiTime(Structure):
    _pack_=2
    _fields_ = [
        ("Seconds", c_short),       # Time seconds.
        ("Minutes", c_short),       # Time minutes.
        ("Hours", c_short),         # Time Hours.
        ("DayOfMonth", c_short),    # Time Day of month.
        ("Month", c_short),         # Time month.
        ("Year", c_short),          # Years Since 1900.
        ("WeekDay", c_short),       # Day since Sunday.
        ("YearDay", c_short),       # Day since January 1st.
    ]


#---
# TMSiDevStorageReport
#---
class TMSiDevStorageReport(Structure):
    _pack_=2
    _fields_ = [
        ("TotalSizeMB", c_uint),    # Total storage in MByte.
        ("UsedSizeMB", c_uint),     # Available storage in MByte.
    ]


#---
# TMSiDevGetConfig
#---
class TMSiDevGetConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),         # The DR serial number.
        ("DRDevID", c_ushort),              # The DR Device ID.
        ("NrOfHWChannels", c_ushort),       # Total nr of hardware channels (UNI, Bip, Aux).
        ("NrOfChannels", c_ushort),         # Total number of hardware + software channels.
        ("NrOfSensors", c_ushort),          # Total supported sensor interfaces.
        ("BaseSampleRateHz", c_ushort),     # Current base samplerate
        ("AltBaseSampleRateHz", c_ushort),  # 4096 / 4000 depends on BaseSampleRateHz.
        ("ConfiguredInterface", c_ushort),  # Communication interface on DR used, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
        ("InterFaceBandWidth", c_int),      # Data bandwidth in MB/s for current interface.
        ("TriggersEnabled", c_short),       # 0= disabled 1 = External triggers enabled.
        ("RefMethod", c_short),             # 0= Common reference, 1=average reference.
        ("AutoRefMethod", c_short),         # 0= fixed method, 1= if average -> common reference.
        ("DRSyncOutDiv", c_short),          # BaseSampleRate/SyncOutDiv,  -1 = markerbut.
        ("DRSyncOutDutyCycl", c_short),     # SyncOutv dutycycle.
        ("DSSyncOutDiv", c_short),          # BaseSampleRate/SyncOutDiv,  -1 = markerbut.
        ("DSSyncOutDutyCycl", c_short),     # SyncOutv dutycycle, relative to DR BaseFreq
        ("RepairLogging", c_short),         # 0 Disabled, 1 = BackupLogging enabled AmbRecording is disabled.
        ("AmbRecording", c_short),          # 0 Disabled, 1 = Ambulant Configured/Enabled
        ("AvailableRecordings", c_short),   # Currently stored recordings on device.
        ("DeviceName", c_char * 18),        # Full dev name 17 char string (zero terminated).
    ]


#---
# TMSIDevChDesc
#---
class TMSiDevChDesc(Structure):
    _pack_=2
    _fields_ = [
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor,5=DIGSTAT, 6=SAW.
        ("ChannelFormat", c_ushort),    # 0x00xx Usigned xx bits, 0x01xx signed xx bits
        ("ChanDivider", c_short),       # -1 disabled, else BaseSampleRateHz >> ChanDivider.
        ("ImpDivider", c_short),        # -1 disabled, else BaseSampleRate>>ImpDivider
        ("ChannelBandWidth", c_int),    # Bandwidth (in MB/s) required for transfer from DR to DS, used by bandwidth manager in application software.
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_char * 10),      # Channel Unit, 9 char zero terminated.
        ("DefChanName", c_char * 10),   # Default channel name 9 char zero terminated.
        ("AltChanName", c_char * 10),   # User configurable name 9 char zero terminated.
    ]


#---
# The basic sensor metadata header describing the type of sensor
#---
class SensorDataHeader(Structure):
    _pack_=2
    _fields_ = [
	    ("ManufacturerID", c_ushort),       # Who makes this accessory.
    	("Serialnr", c_uint),               # Serial number.
    	("ProductIdentifier", c_ulonglong), # Together with (AI) and serial number forming UDI
    	("ChannelCount", c_ubyte),          # Indicates the number of channel structs
    	("AdditionalStructs", c_ubyte),     # Indicates the number of additional Structs
    ]


#---
# Sensor Structure [StructID 0x0000] 32 bytes
#---
class SensorDefaultChannel(Structure):
    _pack_=2
    _fields_ = [
    	("StructID", c_ushort),         # ID (= 0x0000)of the channel struct according to XML-file
    	("ChannelName", c_char * 10),   # Zero terminated string for the channel name.
    	("UnitName", c_char * 10),      # Zero terminated string for the Unit string.
    	("Exponent", c_short),          # Exponent for the unit, e.g. milli = -3 this gives for a UnitName V a result mV.
    	("Gain", c_float),              # Value to convert the sensor value to the correct unit value.
    	("Offset", c_float),            # Offset for this channel.
    ]


#---
# Sensor Structure [StructID 0xFFFF] 2 bytes
#---
class SensorDummyChannel(Structure):
    _pack_=2
    _fields_ = [
	    ("StructID", c_ushort), # ID (= 0xFFFF) of the dummy channel struct
    ]

#---
# Sensor Structure [StructID 0x03FE] 74 bytes
#---
class TMSiContact(Structure):
    _pack_=2
    _fields_ = [
    	("StructID", c_ushort),         #  ID (= 0x3FE) of the TMSi Contact struct according to XML-file
    	("CompanyName", c_char * 10),   # Zero terminated string for the Company name.
    	("WWW", c_char * 20),           # Zero terminated string for web URL.
    	("Email", c_char * 30),         # Zero terminated string for support email.
        ("Phone", c_char * 16),         # Zero terminated string for telephone number.
    ]


#---
# TMSiDevSetConfig
#---
class TMSiDevSetConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),             # The DR serial number.
        ("NrOfChannels", c_ushort),             # total nr of channels in this configuration.
        ("SetBaseSampleRateHz", c_ushort),      # New base samplerate for DR.
        ("SetConfiguredInterface", c_ushort),   # Comm interface on DR to configure, 0 = Unknown, 1=USB  2=Network, 3=WiFi, 4=Electrical,5=Optical, 6=Bluetooth.
        ("SetTriggers", c_short),               # 0= Disabled, 1= external triggers enabled.
        ("SetRefMethod", c_short),              # 0= Common reference, 1=average reference.
        ("SetAutoRefMethod", c_short),          # 0= fixed method, 1= if average -> common.
        ("SetDRSyncOutDiv", c_short),           # SetBaseSampleRateHz/SyncOutDiv, -1 = markerbut.
        ("DRSyncOutDutyCycl", c_short),         # Set DR Sync dutycycle.
        ("SetRepairLogging", c_short),          # 0 Disabled, 1 = BackupLogging enabled, Ambulatory recording is disabled!
        ("PerformFactoryReset", c_short),       # Set device to defaults, all other options in this config are ignored by device.
        ("StoreAsDefault", c_short),            # Set the current configuration as default startup configuration.
        ("WebIfCtrl", c_ushort),                # Status of DS WebIF stop = 0, start =1
        ("PinKey", c_byte * 4),                 # Pincode to use for pairing procedure.
    ]


#---
# TMSiDevSetChCfg
#---
class TMSiDevSetChCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),           # Which channel is this configure for.
        ("ChanDivider", c_short),       # -1 disabled, else SetBaseSampleRateHz>>ChanDivider.
        ("AltChanName", c_byte * 10),   # User configurable name 9 char zero terminated.
    ]


#---
# TMSiDevGetSens
#---
class TMSiDevGetSens(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),               # Channel where sensor is connected to.
        ("IOMode", c_short),                # -1 Disabled, 0=Nonin, 1=SPI, 2=UART, 20=Analog
        ("SensorID", c_short),              # ID of connected Sensor to this channel
        ("SensorMetaData", c_byte * 128),   # Additional raw data from sensor.
    ]


#---
# TMSiSetDevSens
#---
class TMSiSetDevSens(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),   # Channel of connected sensor
        ("IOMode", c_short),    # Channel Sensor communication method: -1 Disabled, 0=Nonin, 1=SPI, 2=UART, 20=Analog
    ]


#---
# TMSiDevSampleReq
#---
class TMSiDevSampleReq(Structure):
    _pack_=2
    _fields_ = [
        ("SetSamplingMode", c_ushort),      # flag to start and stop, see SampleControlType commands.
        ("DisableAutoswitch", c_ushort),    # Ignore the Refmode autoswitch for now.
        ("DisableRepairLogging", c_ushort), # Ignore the Repairlogging for now.
        ("DisableAvrRefCalc", c_ushort),    # Disable average ref. calculation for now.
    ]


#---
# TMSiDevImpReq
#---
class TMSiDevImpReq(Structure):
    _pack_=2
    _fields_ = [
        ("SetImpedanceMode", c_ushort), # flag to start and stop, bitmask defined as ImpedanceControlType can be used.
    ]


#---
# TMSiDevRecList
#---
class TMSiDevRecList(Structure):
    _pack_=2
    _fields_ = [
        ("RecFileID", c_ushort),        # Identifier for this file.
        ("RecFileName", c_char * 32),   # Filename
        ("StartTime", TMSiTime),        # StartTime of this recording.
        ("StopTime", TMSiTime),         # StopTime of this recording.
    ]


#---
# TMSiDevRecDetails,
#---
class TMSiDevRecDetails(Structure):
    _pack_=2
    _fields_ = [
        ("StructID", c_short),
        ("ProtoVer", c_short),
        ("RecFileType", c_short),
        ("RecFileID", c_short),
        ("StorageStatus", c_int),
        ("NoOfSamples", c_int),
        ("RecFileName", c_char * 32),
        ("StartTime", TMSiTime),
        ("StopTime", TMSiTime),
        ("ImpAvailable", c_short),
        ("PatientID", c_char * 128),
        ("UserString1", c_char * 64),
        ("UserString2", c_char * 64),
    ]


#---
# TMSiDevImpReport
#---
class TMSiDevImpReport(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_ushort),   # The channel for which this impedance value is.
        ("Impedance", c_float), # The actual impedance value for this channel.
    ]


#---
# TMSiDevRecCfg
#---
class TMSiDevRecCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ProtoVer", c_short),              # Version of the current spec used.
        ("FileType", c_short),              # Type of file set by device.
        ("StartControl", c_short),          # Configuration how to start the ambulant recording.
        ("EndControl", c_int),              # Configuration how to stop the amplulant recording.
        ("StorageStatus", c_int),           # Status of the internal storage.
        ("InitIdentifier", c_int),          # Identifier can be used by the application.
        ("PrefixFileName", c_byte * 16),    # Prefix for the final recording filename.
        ("StartTime", TMSiTime),            # The start time for the recording.
        ("StopTime", TMSiTime),             # The stop time for the recording.
        ("IntervalSeconds", c_short),       # Recuring start time seconds.
        ("IntervalMinutes", c_short),       # Recuring start time minutes.
        ("IntervalHours", c_short),         # Recuring start time hours.
        ("IntervalDays", c_short),          # Recuring start time days.
        ("AlarmTimeCount", c_int),          # Amount of recurring cycles.
        ("PreImp", c_short),                # Pre measurement impedance 0=no, 1=yes.
        ("PreImpSec", c_short),             # Amount of seconds for impedance.
        ("PatientID", c_byte * 128),        # Freeformat string, can be set by application.
        ("UserString1", c_byte * 64),       # Freeformat string, can be set by application.
        ("UserString2", c_byte * 64),       # Freeformat string, can be set by application.
    ]


#---
# TMSiDevRepairReq
#---
class TMSiDevRepairReq(Structure):
    _pack_=2
    _fields_ = [
        ("SampleStartCntr", c_uint),    # Sample Saw counter Start.
        ("NROfSampleSets", c_uint),     # Number of sets to retrieve.
    ]


#---
# TMSiDevChCal
#---
class TMSiDevChCal(Structure):
    _pack_=2
    _fields_ = [
        ("ChanNr", c_uint),             # Which channel is this configure for.
        ("ChanGainCorr", c_float),      # A float value for the Gain calibration.
        ("ChanOffsetCorr", c_float),    # A float value for the Offset calibration.
    ]


#---
# TMSiDevGetDiagStat
#---
class TMSiDevGetDiagStat(Structure):
    _pack_=2
    _fields_ = [
        ("DRHealthState", c_ushort),    # Current state of device since last boot, 0 = OK, 1 = Error.
        ("DRErrors", c_short),          # Nr of Errors logged since logging was started, -1 logging disabled.
        ("DRLogSize", c_uint),          # DR Log size in Bytes.
        ("DSHealthState", c_ushort),    # Current state of device since last boot, 0 = OK, 1 = Error.
        ("DSErrors", c_short),          # Nr of Errors logged since logging was started, -1 logging disabled.
        ("DSLogSize", c_uint),          # DS Log size in Bytes.
    ]


#---
# TMSiDevSetDiagStat
#---
class TMSiDevSetDiagStat(Structure):
    _pack_=2
    _fields_ = [
        ("DRLoggingState", c_ushort),   # Set logging state of device, 0=disable, 1= enable.
        ("DRResetLog", c_ushort),       # Reset the logfile, old loggings are erased.
        ("DSLoggingState", c_ushort),   # Set logging state of device, 0=disable, 1= enable.
        ("DSResetLog", c_ushort),       # Reset the logfile, old loggings are erased.
    ]


#---
# TMSiDevFWStatusReport
#---
class TMSiDevFWStatusReport(Structure):
    _pack_=2
    _fields_ = [
        ("FWVersion", c_short),     # -1 N.A., Vaa.bb  -> 0xaabb
        ("AppVersion", c_short),    # -1 N.A., Vaa.bb  -> 0xaabb
        ("FWStatus", c_int),        # -1 Unknown, All_OK, Upgrading, Verify_OK, Verify_Fail,
        ("MaxPushSize", c_uint),    # max allowed block size in bytes when sending firmware to device.
    ]


#---
# TMSiFWHeaderFile
#---
class TMSiFWHeaderFile(Structure):
    _pack_=2
    _fields_ = [
        ("FWVersion", c_short),         # The Firmware version of this file.
        ("FWHardwareVersion", c_short), # The hardware version for this firmware 0xVVRR.
        ("DevID", c_short),             # The DevID for which this firmware is intended.
        ("FWSize", c_uint),             # The total size in bytes of the firmware to flash.
        ("Checksum", c_int),            # Integrity check type t.b.d.
    ]


#---
# TMSiDevProductConfig
#---
class TMSiDevProductConfig(Structure):
    _pack_=2
    _fields_ = [
        ("DRSerialNumber", c_uint),     # The DR serial number.
        ("DSSerialNumber", c_uint),     # The DS serial number.
        ("DRDevID", c_ushort),          # DR Device ID
        ("DSDevID", c_ushort),          # DS Device ID
        ("NrOfHWChannels", c_ushort),   # total nr of UNI, Bip, Aux channels.
        ("NrOfChannels", c_ushort),     # Total number of channels.
    ]


#---
# TMSiDevProductChCfg
#---
class TMSiDevProductChCfg(Structure):
    _pack_=2
    _fields_ = [
        ("ChannelType", c_ushort),      # 0=Unknown, 1=UNI, 2=BIP, 3=AUX, 4=DIGRAW/Sensor,5=DIGSTAT, 6=SAW.
        ("ChannelFormat", c_ushort),    # 0x00xx Usigned xx bits, 0x01xx signed xx bits
        ("Unitconva", c_float),         # Unit = a*Bits + b used for bits -> unit, IEEE 754
        ("Unitconvb", c_float),
        ("Exp", c_short),               # Exponent, 3= kilo,  -6 = micro etc.
        ("UnitName", c_byte * 10),      # Channel Unit, 9 char zero terminated.
        ("DefChanName", c_byte * 10),   # Default channel name 9 char zero terminated.
    ]


#---
# TMSiDevNetworkConfig
#---
class TMSiDevNetworkConfig(Structure):
    _pack_=2
    _fields_ = [
        ("NetworkMode", c_ushort),      # 0 = Network disabled, 1 = Use DHCP, 2 = Use config as below.
        ("DSIPAddress", c_byte * 16),   # Static DS IP Address.
        ("DSNetmask", c_byte * 16),     # Static DS Netmask.
        ("DSGateway", c_byte * 16),     # Static DS Gateway Address.
    ]


#---
# TMSiDevCardStatus
#---
class SagaCardStatus(Structure):
    _pack_=1
    _fields_ = [
        ("NrOfRecordings", c_ushort),   # Number of available card recordings.
        ("TotalSpace", c_uint),         # Total space in MB
        ("AvailableSpace", c_uint),     # Free space in MB
    ]
