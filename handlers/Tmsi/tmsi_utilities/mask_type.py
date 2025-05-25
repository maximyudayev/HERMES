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
 * @file mask_type.py 
 * @brief 
 * Class to handle mask types
 */


'''

trigger_bits = 0b00000000000000000000000011111111 #bits which contain actual trigger data

def robust_reverse(x):
    """Robust function to revert a list or a tuple given as input

    :param x: a list or a numpy array containing the signal to reverse.
    :type x: list or tuple
    :return: list or tuple reverted. if the wrong datatype is provided, the original input is returned.
    :rtype: list or tuple
    """
    
    
    try:
        if isinstance(x, list):
            return [(~int(i) & trigger_bits) for i in x]
        if isinstance(x, tuple):
            return ((~int(i) & trigger_bits) for i in x)
        return x
    except:
        return x

class MaskType():
    """
    This class contains all the allowed masks.
    """
    DEFAULT = lambda x: x
    REVERSE = robust_reverse