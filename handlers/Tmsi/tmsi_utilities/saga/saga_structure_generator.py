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
 * @file saga_structure_generator.py 
 * @brief 
 * Object that creates the structures needed to communicate with Saga.
 */


'''

import datetime

from ...device.devices.saga import saga_API_structures as SagaStructures
from ...device.devices.saga import saga_API_enums as SagaEnums
from ...device.devices.saga.saga_device import SagaDevice


class SagaStructureGenerator:
    """Class to handle the generation of structures useful for the SagaDevice"""
    def create_card_record_configuration(
        device: SagaDevice,
        start_control: SagaEnums.SagaStartCardRecording = None,
        prefix_file_name: str = None,
        start_time: datetime.datetime = None,
        stop_time: datetime.datetime = None,
        pre_measurement_imp = None,
        pre_measeurement_imp_seconds = None,
        user_string_1: str = None,
        user_string_2: str = None,
        patient_id: str = None
    ) -> SagaStructures.TMSiDevRecCfg:
        """Creates the TMSiDevRecCfg structure with provided parameters

        :param device: device to pull the configuration from
        :type device: SagaDevice
        :param start_control: how to start recording, defaults to None.
        :type start_control: SagaEnums.SagaStartCardRecording, optional
        :param prefix_file_name: prefix file name, defaults to None.
        :type prefix_file_name: str, optional
        :param start_time: datetime of the start, defaults to None.
        :type start_time: datetime.datetime, optional
        :param stop_time: datetime of the stop, defaults to None.
        :type stop_time: datetime.datetime, optional
        :param user_string_1: user string, defaults to None.
        :type user_string_1: str, optional
        :param user_string_2: user string, defaults to None.
        :type user_string_2: str, optional
        :param patient_id: user string, defaults to None.
        :type patient_id: str, optional
        :return: the structure containing provided information
        :rtype: SagaStructures.TMSiDevRecCfg
        """
        
        config = device.get_card_recording_config()
        if start_control is not None:
            config.StartControl = start_control.value
        if prefix_file_name is not None:
            max_len = len(prefix_file_name)
            prefix_file_name = bytearray(prefix_file_name, 'utf-8')
            converted_str = bytearray(SagaEnums.SagaStringLengths.PrefixFileName.value)
            converted_str[:max_len] = prefix_file_name[:max_len]
            config.PrefixFileName[:] = converted_str
        if user_string_1 is not None:
            max_len = len(user_string_1)
            user_string_1 = bytearray(user_string_1, 'utf-8')
            converted_str = bytearray(SagaEnums.SagaStringLengths.UserString.value)
            converted_str[:max_len] = user_string_1[:max_len]
            config.UserString1[:] = converted_str
        if user_string_2 is not None:
            max_len = len(user_string_2)
            user_string_2 = bytearray(user_string_2, 'utf-8')
            converted_str = bytearray(SagaEnums.SagaStringLengths.UserString.value)
            converted_str[:max_len] = user_string_2[:max_len]
            config.UserString2[:] = converted_str
        if patient_id is not None:
            max_len = len(patient_id)
            patient_id = bytearray(patient_id, 'utf-8')
            converted_str = bytearray(SagaEnums.SagaStringLengths.PatientString.value)
            converted_str[:max_len] = patient_id[:max_len]
            config.PatientID[:] = converted_str
        if start_time is not None:
            SagaStructureGenerator.from_datetime_to_tmsitime(
                start_time, config.StartTime)
        if stop_time is not None:
            SagaStructureGenerator.from_datetime_to_tmsitime(
                stop_time, config.StopTime)
        if pre_measurement_imp is not None:
            config.PreImp = pre_measurement_imp
        if pre_measeurement_imp_seconds is not None:
            config.PreImpSec = pre_measeurement_imp_seconds
        return config


    def from_qdatetime_to_tmsitime(qdatetime, tmsi_time):
        """Convert QDateTime to TMSiTime.

        :param qdatetime: QDateTime.
        :type qdatetime: QDateTime
        :param tmsi_time: TMSiTime.
        :type tmsi_time: TMSiTime
        :return: TMSiTime.
        :rtype: TMSiTime
        """
        tmsi_time.Seconds = qdatetime.time().second()
        tmsi_time.Minutes = qdatetime.time().minute()
        tmsi_time.Hours = qdatetime.time().hour()
        tmsi_time.DayOfMonth = qdatetime.date().day()
        tmsi_time.Month = qdatetime.date().month() - 1
        tmsi_time.Year = qdatetime.date().year()-1900
        return tmsi_time

    def from_datetime_to_tmsitime(date_time, tmsi_time):
        """Convert datetime to TMSiTime.

        :param date_time: datetime.
        :type date_time: datetime
        :param tmsi_time: TMSiTime.
        :type tmsi_time: TMSiTime
        :return: TMSiTime.
        :rtype: TMSiTime
        """
        tmsi_time.Seconds = date_time.time().second
        tmsi_time.Minutes = date_time.time().minute
        tmsi_time.Hours = date_time.time().hour
        tmsi_time.DayOfMonth = date_time.date().day
        tmsi_time.Month = date_time.date().month - 1
        tmsi_time.Year = date_time.date().year-1900
        return tmsi_time

    
    def from_tmsitime_to_datetime(tmsi_time, date_time):
        """Convert TMSiTime to datetime.

        :param tmsi_time: TMSiTime.
        :type tmsi_time: TMSiTime
        :param date_time: datetime.
        :type date_time: datetime
        :return: datetime.
        :rtype: datetime
        """
        date_time = datetime.datetime(
            tmsi_time.Year + 1900,
            tmsi_time.Month + 1,
            tmsi_time.DayOfMonth,
            tmsi_time.Hours,
            tmsi_time.Minutes,
            tmsi_time.Seconds)
        return date_time

    