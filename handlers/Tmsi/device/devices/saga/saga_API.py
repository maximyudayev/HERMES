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
 * @file saga_API.py 
 * @brief 
 * API calls to communication library.
 */


'''

from ctypes import *
from sys import platform
import os
from array import *

from .saga_API_enums import *
from .saga_API_structures import *
from ....tmsi_utilities.tmsi_logger import TMSiLogger

DeviceHandle = c_void_p
TMSiDeviceHandle = DeviceHandle(0)
SagaDllAvailable = False
SagaDllLocked = True


if platform == "linux" or platform == "linux2":

    so_name = "libTMSiSagaDeviceLib.so"
    soabspath = os.path.sep + os.path.join('usr', 'lib', so_name)
    dlclose_func = cdll.LoadLibrary('').dlclose
    dlclose_func.argtypes = [c_void_p]

    try:
        CDLL("librt.so.1",  RTLD_GLOBAL)
        SagaSDK = CDLL(soabspath,  RTLD_GLOBAL)
        sdk_handle = SagaSDK._handle
        TMSiLogger().debug("Successfully loaded SAGA device library, handle: " + hex(sdk_handle) )
        SagaDllAvailable = True
        SagaDllLocked = False
    except Exception as e:
        TMSiLogger().warning(e)
elif platform == "win32": # Windows
    search_path = "C:/Program files/TMSi/Saga"
    name = "TMSiSagaDeviceLib.dll"
    result = os.path.join(search_path, name)
    so_name = os.path.abspath(result)
    if os.path.exists(so_name):
        TMSiLogger().debug("{} available.".format(so_name))
        SagaDllAvailable = True
    try:
        SagaSDK = CDLL(so_name)
        SagaDllLocked = False
        sdk_handle = SagaSDK._handle
        TMSiLogger().debug("Successfully loaded SAGA device library, handle: " + hex(sdk_handle) )
    except Exception as e:
        if SagaDllAvailable:
            TMSiLogger().warning("{} already in use.".format(so_name))
else:
    TMSiLogger().warning("Unsupported platform")


if SagaDllAvailable and not SagaDllLocked:
    # DLL interface

    #---
    # @details This command is used to retrieve a list of available TMSi devices
    # connected to the PC. This query is performed on the "DSInterfaceType" specified
    # by the user. All other interface types are ignored. For each device found on
    # the PC matching "DSInterfaceType" the appropriate low level command is send with
    # "DRInterfaceType" set.
    #
    # @Pre \ref No device should have been opened. Device is in Close state.
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0101.
    #
    # @param[out] TMSiDeviceList  List of found devices.
    # @param[in] DSInterfaceType Interface to DS to query. 0 = Unknown,1=USB,
    # 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @param[in] DRInterfaceType Interface to DR to query. 0 = Unknown,1=USB,
    # 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceList = SagaSDK.TMSiGetDeviceList
    TMSiGetDeviceList.restype = TMSiDeviceRetVal
    TMSiGetDeviceList.argtype = [POINTER(TMSiDevList), c_int, c_uint, c_uint]

    #---
    # @details This command is used to open a device. This will create a connection
    # between API and DS and "lock" the interface between DR and DS.
    #
    # @Pre @li No device should have been openend.
    # @li TMSiGetDeviceList should have been called to obtain valid
    #       TMSiDeviceID
    #
    # @Post After TMSI_OK device is in "Device_Open".
    #
    # @depends Low level call 0x0102.
    #
    # @param[out] TMSiDeviceHandle  Handle to device use for further API calls.
    # @param[in] DeviceID Device to open, retrieved by "TMSiGetDeviceList".
    # @param[in] DRInterfaceType Interface to DR to use and lock. 0 = Unknown,1=USB, 2=Network, 3=WiFi, 4=Electrical, 5=Optical, 6=Bluetooth.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiOpenDevice = SagaSDK.TMSiOpenDevice
    TMSiOpenDevice.restype = TMSiDeviceRetVal
    TMSiOpenDevice.argtype = [POINTER(c_void_p), c_uint, c_uint]


    #---
    # @details This command is used to Close a device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post After TMSI_OK the device STATEMACHINE is in "Device_Close" state.
    #
    # @depends Low level call 0x0103.
    #
    # @param[out] TMSiDeviceHandle  Handle of device to close.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiCloseDevice = SagaSDK.TMSiCloseDevice
    TMSiCloseDevice.restype = TMSiDeviceRetVal
    TMSiCloseDevice.argtype = [c_void_p]


    #---
    # @details This command is used to retrieve a status report from a TMSi device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # @Post No device change.
    #
    # @depends Low level call 0x0201.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] DeviceStatus      Status report of the connected device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceStatus = SagaSDK.TMSiGetDeviceStatus
    TMSiGetDeviceStatus.restype = TMSiDeviceRetVal
    TMSiGetDeviceStatus.argtype = [c_void_p, POINTER(TMSiDevStatReport)]

    #---
    # @details This command is used to retrieve a full status report from a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0202.
    #
    # @param[in] TMSiDeviceHandle   Handle to the current open device.
    # @param[out] FullDeviceStatus  Status report.
    # @param[out] DeviceBatteryReportList   list of BatteryReport(s).
    # @param[in] BatteryStatusListLen   Nr of BatteryReportLists allocated.
    #
    # @param[out] DeviceTime Device time report.
    # @param[out] StorageReport Device storage report.
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetFullDeviceStatus(void* TMSiDeviceHandle, TMSiDevFullStatReportType* FullDeviceStatus, TMSiDevBatReportType* DeviceBatteryStatusList, int32_t BatteryStatusListLen, TMSiTimeType* DeviceTime, TMSiDevStorageReportType* StorageReport);
    TMSiGetFullDeviceStatus = SagaSDK.TMSiGetFullDeviceStatus
    TMSiGetFullDeviceStatus.restype = TMSiDeviceRetVal
    TMSiGetFullDeviceStatus.argtype = [c_void_p, POINTER(TMSiDevFullStatReport), POINTER(TMSiDevBatReport), c_int, POINTER(TMSiTime), POINTER(TMSiDevStorageReport)]

    #---
    # @details This command is used to retrieve the current configuration from a
    # TMSi device. The response can be used to calculate the expected data streams
    # from the device. If a channel is enabled for sampling the "ChanDivider > -1".
    # If a channel will send its impedance data during an impedance measurement is
    # determined by the "ImpDivider".  The last channel in the device list (after
    # the SAW channel) is the PGND, this channel is included only for use during
    # impedance mode. Therefore the ChanDivider = -1 and the ImpDivider is > -1 for
    # this channel. To calculate the expected data stream in a certain mode multiply
    # all enabled channels with the sample frequency and int32_t.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No device change.
    #
    # @depends Low level call 0x0203.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RecorderConfiguration Device configuration.
    # @param[out] ChannelsList  Channel(s) configuration.
    # @param[in] ChannelsListLen    Allocated nr of ChannelList items,
    # should be atleast NrOfChannels as mentioned in TMSiDevStatReportType
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceConfig = SagaSDK.TMSiGetDeviceConfig
    TMSiGetDeviceConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceConfig.argtype = [c_void_p, POINTER(TMSiDevGetConfig), POINTER(TMSiDevChDesc), c_int]

    #---
    # @details This command is used to set a new configuration on a TMSi
    # device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When a TMSI_OK is returned the Device configuration is updated. When an # error is returned the device configuration might not be updated. This should
    # be checked by requesting the current configuration \ref TMSiGetDeviceConfig.
    #
    # @depends Low level call 0x0204.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] RecorderConfiguration  Buffer with new configuration.
    # @param[in] ChannelConfigList      Buffer with ChannelConfigList items.
    # @param[in] ChannelConfigListLen   Nr of ChannelConfigList items.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceConfig(void* TMSiDeviceHandle, TMSiDevSetConfigType* RecorderConfiguration, TMSiDevSetChCfgType* ChannelConfigList, int32_t ChannelConfigListLen);
    TMSiSetDeviceConfig = SagaSDK.TMSiSetDeviceConfig
    TMSiSetDeviceConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceConfig.argtype = [c_void_p, POINTER(TMSiDevSetConfig), POINTER(TMSiDevSetChCfg), c_int]

    #---
    # @details This command is used to set the time on a TMSi device.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When a TMSI_OK is returned the internal time has been updated.
    #
    # @depends Low level call 0x0205.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] NewTime    Buffer with new time information.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceRTC(void* TMSiDeviceHandle, TMSiTimeType* NewTime);
    TMSiSetDeviceRTC = SagaSDK.TMSiSetDeviceRTC
    TMSiSetDeviceRTC.restype = TMSiDeviceRetVal
    TMSiSetDeviceRTC.argtype = [c_void_p, POINTER(TMSiTime)]

    #---
    # @details This command is used to get sensor information from channels which
    # support this feature.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0206.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] SensorList    Channel(s) configuration.
    #
    # @param[in] SensorsListLen Nr of SensorList elements allocated, should be
    # atleast NrOfSensors from TMSiDevGetConfigType.
    # @param[out] RetSensorsListLen  Nr of Sensor lists returned.
    #                   .
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceSensor(void* TMSiDeviceHandle, TMSiDevGetSensType* SensorsList, uint32_t SensorsListLen, uint32_t* RetSensorsListLen);
    TMSiGetDeviceSensor = SagaSDK.TMSiGetDeviceSensor
    TMSiGetDeviceSensor.restype = TMSiDeviceRetVal
    TMSiGetDeviceSensor.argtype = [c_void_p, POINTER(TMSiDevGetSens), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to set sensor options for channels which
    # support this feature.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post When TMSI_OK returned the device sensor configuration is as requested,
    # else the previous configuration is still valid.
    #
    # @depends Low level call 0x0207.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] SensorList    List of sensor configuration(s).
    # @param[out] SensorsListLen    Nr of SensorList items.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceSensor(void* TMSiDeviceHandle, TMSiSetDevSensType* SensorList, int32_t SensorsListLen);

    #---
    # @details This command is used to control the sampling mode on a TMSi
    # device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Sampling".
    #
    # @Post When TMSI_OK is returned the device is in the "Device_Sampling" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Sampling data should be retrieved by calling TMSiGetDeviceData.
    # STOPWiFiStream will result in a stop of datastream the device will go to
    # ambulant recording mode, the connection to TMSiDevice shall be closed by the
    # application.
    #
    # @depends Low level call 0x0301.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceSamplingMode New device sampling configuration.
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiSetDeviceSampling = SagaSDK.TMSiSetDeviceSampling
    TMSiSetDeviceSampling.restype = TMSiDeviceRetVal
    TMSiSetDeviceSampling.argtype = [c_void_p, POINTER(TMSiDevSampleReq)]

    #---
    # @details This command is used to set the device impedance mode.
    #
    # @Pre \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Impedance".
    #
    # @Post When TMSI_OK is returned the device is in the "Device_Impedance" or
    # "Device_Open" state depending on the requested StartStop flag.
    # Impedance data should be retrieved by calling TMSiGetDeviceData.
    #
    # @depends Low level call 0x0302.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceImpedanceMode    New device impedance configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceImpedance(void* TMSiDeviceHandle, TMSiDevImpReqType* DeviceImpedanceMode);
    TMSiSetDeviceImpedance = SagaSDK.TMSiSetDeviceImpedance
    TMSiSetDeviceImpedance.restype = TMSiDeviceRetVal
    TMSiSetDeviceImpedance.argtype = [c_void_p, POINTER(TMSiDevImpReq)]

    #---
    # @details This command is used to get the device streaming data. The
    # application can retrieve sampledata/impdata from the device. It returns data
    # as 32-bit float values, all data is already processed, meaning it is converted
    # from bits to units (as specified in the channel descriptor). The function will # return a buffer with a NrOfSets of samples, for each ENABLED channel one
    # sample per set. The application should match each sample with the
    # corresponding channel. All samples are in order of enabled channels starting
    # at the first channel.
    # The DataType indicates if the data is Sampledata DataType = 1,  ImpedanceData
    # DataType = 2, Sampledata Recording = 3.
    # In case of impedance data only Channels with "ImpDivider" > -1 are transmitted.
    # The buffer retured is a multiple of Samplesets.
    #
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle. The device shall be in "Device_Sampling" or
    # "Device_Impedance" state.
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0303.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceData       Received device Data.
    # @param[in] DeviceDataBufferSize      Buffersize for device Data;
    # @param[out] NrOfSets     The returned samplesets in this buffer
    # @param[out] DataType     The returned data type.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    TMSiGetDeviceData = SagaSDK.TMSiGetDeviceData
    TMSiGetDeviceData.restype = TMSiDeviceRetVal
    TMSiGetDeviceData.argtype = [c_void_p, POINTER(c_float), c_uint, POINTER(c_uint), POINTER(c_int)]

    #---
    # @details This command is used to get the current status of the streaming
    # databuffer. It returns the current value of the amount of data waiting in the
    # buffer.
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle. The device shall be in "Device_Sampling" or
    # "Device_Impedance" state.
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceDataBuffered  The amount of data buffered for this device in
    # Bytes.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceDataBuffered(void* TMSiDeviceHandle, int32_t* DeviceDataBuffered);

    #---
    # @details This command is used to reset the internal data buffer thread for the
    # specified device after it has been stopped sampling.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends None, API call only.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DLL error received.
    #---
    TMSiResetDeviceDataBuffer = SagaSDK.TMSiResetDeviceDataBuffer
    TMSiResetDeviceDataBuffer.restype = TMSiDeviceRetVal
    TMSiResetDeviceDataBuffer.argtype = [c_void_p]

    #---
    # @details This command is used to get the device storage list.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0304.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RecordingsList   List of available recordings on data recorder.
    # @param[in] RecordingsListLen     Buffersize for RecordingsList
    # @param[out] RetRecordingListLen   The amount of returned recordings in the list.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---

    TMSiGetDeviceStorageList = SagaSDK.TMSiGetDeviceStorageList
    TMSiGetDeviceStorageList.restype = TMSiDeviceRetVal
    TMSiGetDeviceStorageList.argtype = [c_void_p, POINTER(TMSiDevRecList), c_uint, POINTER(c_uint)]

    #---
    # @details This command is used to get a recorded file from the data recorder
    # device. The file is selected by using the "RecFileID" as returned by
    # "TMSiGetDeviceStorageList". After a successful return from this call the file
    # sample data can be retrieved by calling the "TMSiGetDeviceData" where the
    # "DataType" flag will be 3.  To Stop / abort the file transfer this call can be
    # used with the apropriate StartStop flag.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Amb_Data".
    #
    # @Post If TMSI_OK the device STATEMACHINE shall be in "Device_Open" or
    # "Device_Amb_Data".
    #
    # @depends Low level call 0x0305.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] RecFileID     The file ID which should be retrieved.
    # @param[in] StartStop;  flag to start (1) and stop (0) the file transfer.
    # @param[out] RecordingMetaData Metadata of requested recording file.
    # @param[out] ImpedanceReportList      An impedance value for each impedance
    # enabled channel.
    # @param[in] ImpedanceReportListLen   Size of mpedanceReportListshould be
    # atleast the size of all channels with ImpDivider > -1 * int32_t. ImpDivider is
    # found in TMSiDevChDescType.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetRecordingFile(void* TMSiDeviceHandle, uint16_t RecFileID, uint16_t StartStop, TMSiDevRecDetailsType* RecordingMetaData, TMSiDevImpReportType* ImpedanceReportList, int32_t ImpedanceReportListLen);

    TMSiGetRecordingFile = SagaSDK.TMSiGetRecordingFile
    TMSiGetRecordingFile.restype = TMSiDeviceRetVal
    TMSiGetRecordingFile.argtype = [c_void_p, c_ushort, c_ushort, POINTER(TMSiDevRecDetails), POINTER(TMSiDevImpReport), c_uint]

    #---
    # @details This command is used to get a ambulant recording configuration from
    # the data recorder device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open".
    # *
    # @Post No change in device state.
    #
    # @depends Low level call 0x0306.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] AmbulantConfiguration    The current ambulant configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceAmbConfig(void* TMSiDeviceHandle, TMSiDevRecCfgType* AmbulantConfiguration);

    #---

    TMSiGetDeviceAmbConfig = SagaSDK.TMSiGetDeviceAmbConfig
    TMSiGetDeviceAmbConfig.restype = TMSiDeviceRetVal
    TMSiGetDeviceAmbConfig.argtype = [c_void_p, POINTER(TMSiDevRecCfg)]

    #---
    # @details This command is used to set a new ambulant recording configuration
    # from the data recorder device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0307.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] AmbulantConfiguration    The new ambulant configuration.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal  TMSiSetDeviceAmbConfig(void* TMSiDeviceHandle, TMSiDevRecCfgType* AmbulantConfiguration);

    #---

    TMSiSetDeviceAmbConfig = SagaSDK.TMSiSetDeviceAmbConfig
    TMSiSetDeviceAmbConfig.restype = TMSiDeviceRetVal
    TMSiSetDeviceAmbConfig.argtype = [c_void_p, POINTER(TMSiDevRecCfg)]

    #---
    # @details This command is used to get repair data from a device after a measurement.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li The device STATEMACHINE shall be in "Device_Open" or "Device_Repair".
    # @Post No change in device state.
    #
    # @depends Low level call 0x0308.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] RepairDataBuffer     Buffer to copy the requested repairdata into.
    # @param[in] RepairDataBufferSize   The available buffersize in bytes.
    # @param[out] NrOfSamples  The returned number of samples.
    # @param[in] RepairInfo The repair request
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal  TMSiGetDeviceRepairData(void* TMSiDeviceHandle, float* RepairDataBuffer, int32_t RepairDataBufferSize, int32_t* NrOfSamples, TMSiDevRepairReqType* RepairInfo);

    #---
    # @details This command is used to get calibration data from a device. All
    # available channels will be returned.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    # @Post No change in device state.
    #
    # @depends Low level call 0x0601.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] ChCalValuesList      The list of calibration data.
    # @param[in] ChCalValuesListLen    The amount of allocatedlist items.
    # @param[out] RetChCalValuesListLen    The returned list size.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceCalibration(void* TMSiDeviceHandle, TMSiDevChCalType* ChCalValuesList, int32_t ChCalValuesListLen, int32_t* RetChCalValuesListLen);

    #---
    # @details This command is used to set calibration data for a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0602.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] ChCalValuesList       The list of new calibration data.
    # @param[in] ChCalValuesListLen    The size of ChCalValueList.
    #
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceCalibration(void* TMSiDeviceHandle, TMSiDevChCalType* ChCalValuesList, int32_t ChCalValuesListLen);

    #---
    # @details This command is used to set calibration mode for a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post Device is in "Device_Calibration".
    #
    # @depends Low level call 0x0603.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] SetCalibrationMode    The Calibration mode setting.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceCalibrationMode(void* TMSiDeviceHandle, int32_t SetCalibrationMode);

    #---
    # @details This command is used to get the current logging activity and health state of the device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0604.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] DeviceDiagnostics The short health status report and current logging settings of the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceDiagnostics(void* TMSiDeviceHandle, TMSiDevGetDiagStatType* DeviceDiagnostics);

    #---
    # @details This command is used to set the logging options for a device. When
    # logging is enabled without a logfile reset, the device will append to an
    # existing log file.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0605.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] DeviceDiagnosticsCfg The short health status report of the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceDiagnostics(void* TMSiDeviceHandle, TMSiDevSetDiagStatType* DeviceDiagnosticsCfg);

    #---
    # @details This command is used to get the current logfile of a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0606.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device of which the log file is requested.
    # @param[in] DeviceLogBufferSize The size of the DeviceLogData buffer;
    # @param[out] RetDeviceLogBufferSize The size of the returned log
    # @param[out] DeviceLogData The logdata from the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceLog(void* TMSiDeviceHandle, uint32_t TMSiDevice, uint32_t DeviceLogBufferSize, uint32_t* RetDeviceLogBufferSize, uint8_t* DeviceLogData);

    #---
    # @details This command is used to get the current firmware status of a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0607.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device of which the firmware status is requested.
    # @param[out] FWReport The current firmware status report.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDeviceFWStatus(void* TMSiDeviceHandle, uint32_t TMSiDevice, TMSiDevFWStatusReportType* FWReport);

    #---
    # @details This command is used to prepare a device for a firmware update. This
    # call sends the firmware header which is checked for compatibility.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0608.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device which needs to prepare for a firmware update.
    # @param[in] NewFWHeader The new firmware header for the firmware.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful, firmware header is accepted.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDeviceFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, TMSiFWHeaderFileType* NewFWHeader);

    #---
    # @details This command is used to send the firmware data to a device.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x0609.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The destination device for the firmware update.
    # @param[in] FWDataSize The size of the firmware data in bytes.
    # @param[in] FWData The new firmware data.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiPushFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, uint32_t FWDataSize, uint8_t* FWData);

    #---
    # @details This command is used to initiate or abort a firmware update.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060A.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] TMSiDevice The device which is being updated.
    # @param[in] FWAction The action to perform on the device.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiDoneFWUpdate(void* TMSiDeviceHandle, uint32_t TMSiDevice, int32_t FWAction);

    #---
    # @details This command is used to program production information during
    # manufacturing.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060B.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] ProductConfig The device which is being configured.
    # @param[in] ChannelConfigList The list of channels to configure.
    # @param[in] ChannelConfigListLen The size of the ChannelConfigList.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetProductConfig(void* TMSiDeviceHandle, TMSiDevProductConfigType* ProductConfig, TMSiDevProductChCfgType* ChannelConfigList, uint32_t ChannelConfigListLen);

    #---
    # @details This command is used to set the network configuration for the DS.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060C.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[out] GetDSNetworkConig The current network configuration of the DS.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiGetDevNetworkConfig(void* TMSiDeviceHandle, TMSiDevNetworkConfigType* GetDSNetworkConig);

    #---
    # @details This command is used to set the network configuration for the DS.
    #
    # @Pre @li \ref TMSiOpenDevice should have been called and returned a valid
    # TMSIDeviceHandle.
    # @li device STATEMACHINE shall be in "Device_Open".
    #
    # @Post No change in device state.
    #
    # @depends Low level call 0x060D.
    #
    # @param[in] TMSiDeviceHandle  Handle to the current open device.
    # @param[in] SetDSNetworkConig The new network configuration for the DS.
    #
    # @return
    # @li TMSI_OK Ok, if response received successful.
    # @li Any TMSI_DS*, TMSI_DR*, TMSI_DLL error received.
    #---
    #TMSIDEVICEDLL_API TMSiDeviceRetVal TMSiSetDevNetworkConfig(void* TMSiDeviceHandle, TMSiDevNetworkConfigType* SetDSNetworkConig);



    #endif
