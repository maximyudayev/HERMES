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
 * @file tmsi_device.py 
 * @brief 
 * TMSi Device interface.
 */


'''

class TMSiDevice():
    """A class to handle all the TMSi Devices."""
    def __init__(self):
        """Initialize the device.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def apply_mask(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def close(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def discover(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def download_file_from_device(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def export_configuration(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def import_configuration(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def get_card_recording_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def get_card_status(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_active_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_active_impedance_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_card_file_info(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_bandwidth(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_card_file_list(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_data(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_handle_value(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_impedance_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

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

    def get_device_interface(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_interface_status(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_list(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_name(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_power_status(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_references(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_repair_logging(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_sampling_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_sampling_frequency(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_serial_number(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_state(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_sync_out_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def get_device_time(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_triggers(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_device_type(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

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

    def get_dr_interface(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

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

    def get_file_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_id(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_live_impedance(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_num_active_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_num_active_impedance_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_num_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_num_impedance_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def get_pairing_status(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def get_sdk(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
        
    def open(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def pair_device(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reload_device(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reset_device_card(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reset_device_data_buffer(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reset_device_event_buffer(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reset_masks(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def reset_to_factory_default(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_card_recording_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_active_channels(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_backup_logging(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
    
    def set_device_channel_names(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_download_file_request(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_impedance_request(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_interface(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_references(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_repair_logging(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_sampling_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_sampling_request(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_sync_out_config(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_time(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def set_device_triggers(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def start_download_file(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def start_measurement(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def stop_download_file(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')
    
    def stop_measurement(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')

    def user_abort_download(*args, **kwargs):
        """Function to be overridden by the child class.

        :raises NotImplementedError: Must be overridden by the child class.
        """
        raise NotImplementedError('method not available for this device')