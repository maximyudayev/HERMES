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
 * @file logger_filter.py 
 * @brief 
 * Filter to handle the logs.
 */


'''

PERFORMANCE_LOG = 25
ACTIVITY_LOG = 26


class LoggerFilter(object):
    """Class to filter logs."""

    def __init__(self, level):
        """Initialize the log filter.

        :param level: level to show.
        :type level: int
        """
        self.__level = level
    def filter(self, log_record):
        """Filter the log.

        :param log_record: log to filter.
        :type log_record: logging.log
        :return: True if the log level is correct, False otherwise.
        :rtype: bool
        """
        return log_record.levelno == self.__level
