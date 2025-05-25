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
 * @file saga_sensor.py 
 * @brief 
 * SAGA Sensor object.
 */


'''

class SagaSensor:
    def __init__(self):
        """Initialize Saga Sensor
        """
        self.__idx_total_channel_list = -1
        self.__id = -1
        self.__manufacturer_id = 0
        self.__serial_nr = 0
        self.__product_id = 0
        self.__offset = 0
        self.__gain = 1
        self.__exp = 0
        self.__name = ""
        self.__unit_name = ""

    def get_sensor_name(self):
        """Get sensor name

        :return: sensor name
        :rtype: str
        """
        return self.__name
    
    def get_sensor_exp(self):
        """Get sensor exp

        :return: sensor exp
        :rtype: float
        """
        return self.__exp
    
    def get_sensor_unit_name(self):
        """Get sensor unit_name

        :return: sensor unit_name
        :rtype: str
        """
        return self.__unit_name
    
    def get_sensor_gain(self):
        """Get sensor gain

        :return: sensor gain
        :rtype: float
        """
        return self.__gain
    
    def get_sensor_idx_total_channel_list(self):
        """Get sensor idx_total_channel_list

        :return: sensor idx_total_channel_list
        :rtype: float
        """
        return self.__idx_total_channel_list
    
    def get_sensor_id(self):
        """Get sensor id

        :return: sensor id
        :rtype: float
        """
        return self.__id
    
    def get_sensor_IOMode(self):
        """Get sensor IOMode

        :return: sensor IOMode
        :rtype: float
        """
        return self.__IOMode
    
    def get_sensor_manufacturer_id(self):
        """Get sensor manufacturer_id

        :return: sensor manufacturer_id
        :rtype: float
        """
        return self.__manufacturer_id
    
    def get_sensor_serial_nr(self):
        """Get sensor serial_nr

        :return: sensor serial_nr
        :rtype: float
        """
        return self.__serial_nr
    
    def get_sensor_product_id(self):
        """Get sensor product_id

        :return: sensor product_id
        :rtype: float
        """
        return self.__product_id
    
    def get_sensor_offset(self):
        """Get sensor offset

        :return: sensor offset
        :rtype: float
        """
        return self.__offset
    
    def set_sensor_name(self, name):
        """Set sensor name

        :param name: sensor name
        :type name: str
        """
        new_name = bytearray()
        for i in range (len(name)):
            if name[i] > 127:
                new_name.append(194) #0xC2
            new_name.append(name[i])
        if len(new_name) > 0:
            self.__name = new_name.decode('utf8').rstrip('\x00')
    
    def set_sensor_exp(self, exp):
        """Set sensor exp

        :param exp: sensor exp
        :type exp: float
        """
        self.__exp = exp
    
    def set_sensor_unit_name(self, unit_name):
        """Set sensor unit name

        :param unit_name: sensor unit name
        :type unit_name: str
        """
        new_name = bytearray()
        for i in range (len(unit_name)):
            if unit_name[i] > 127:
                new_name.append(194) #0xC2
            new_name.append(unit_name[i])
        if len(new_name) > 0:
            self.__unit_name = new_name.decode('utf8').rstrip('\x00')
    
    def set_sensor_gain(self, gain):
        """Set sensor gain

        :param gain: sensor gain
        :type gain: float
        """
        self.__gain = gain
    
    def set_sensor_idx_total_channel_list(self, idx_total_channel_list):
        """Set sensor idx_total_channel_list

        :param idx_total_channel_list: sensor idx_total_channel_list
        :type idx_total_channel_list: float
        """
        self.__idx_total_channel_list = idx_total_channel_list
    
    def set_sensor_id(self, id):
        """Set sensor id

        :param id: sensor id
        :type id: float
        """
        self.__id = id
    
    def set_sensor_IOMode(self, IOMode):
        """Set sensor IOMode

        :param IOMode: sensor IOMode
        :type IOMode: float
        """
        self.__IOMode = IOMode
    
    def set_sensor_manufacturer_id(self, manufacturer_id):
        """Set sensor manufacturer_id

        :param manufacturer_id: sensor manufacturer_id
        :type manufacturer_id: float
        """
        self.__manufacturer_id = manufacturer_id
    
    def set_sensor_serial_nr(self, serial_nr):
        """Set sensor serial_nr

        :param serial_nr: sensor serial_nr
        :type serial_nr: float
        """
        self.__serial_nr = serial_nr
    
    def set_sensor_product_id(self, product_id):
        """Set sensor product_id

        :param product_id: sensor product_id
        :type product_id: float
        """
        self.__product_id = product_id
    
    def set_sensor_offset(self, offset):
        """Set sensor offset

        :param offset: sensor offset
        :type offset: float
        """
        self.__offset = offset
    