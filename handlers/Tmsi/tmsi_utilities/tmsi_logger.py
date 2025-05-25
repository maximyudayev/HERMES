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
 * @file tmsi_logger.py 
 * @brief 
 * Loggers used to handle console or file output for informative and debug reasons.
 */


'''

import os
from sys import platform
from datetime import datetime
import logging
from .logger_filter import PERFORMANCE_LOG, LoggerFilter, ACTIVITY_LOG
from .singleton import Singleton
from .support_functions import get_documents_path

logging.addLevelName(PERFORMANCE_LOG, "PERFORMANCE")
logging.addLevelName(ACTIVITY_LOG, "ACTIVITY")
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')


class TMSiLogger(metaclass = Singleton):
    """Class to handle the logs of TMSi."""
    def __init__(self):
        """Initialize the logger."""
        self.__tmsi_log = logging.getLogger("TMSi")
        debug_stream_handler = logging.StreamHandler()
        debug_stream_handler.setFormatter(formatter)
        self.__tmsi_log.handlers = [debug_stream_handler]
    
    def critical(self, message):
        """Write a log with level critical.

        :param message: message to log.
        :type message: str
        """
        self.__tmsi_log.log(level = logging.CRITICAL, msg = message)
    
    def debug(self, message):
        """Write a log with level debug.

        :param message: message to log.
        :type message: str
        """
        self.__tmsi_log.log(level = logging.DEBUG, msg = message)

    def info(self, message):
        """Write a log with level info.

        :param message: message to log.
        :type message: str
        """
        self.__tmsi_log.log(level = logging.INFO, msg = message)
    
    def warning(self, message):
        """Write a log with level warning.

        :param message: message to log.
        :type message: str
        """
        self.__tmsi_log.log(level = logging.WARNING, msg = message)

class TMSiLoggerActivity(metaclass = Singleton):
    """Class to handle the activity logs."""
    def __init__(self):
        """Initialize the activity logger."""
        self.activity_log_enabled = False
        if "TMSi_ACTIVITY" not in os.environ:
            return
        if os.environ["TMSi_ACTIVITY"] != "ON":
            return
        self.activity_log_enabled = True
        self.__tmsi_perf = logging.getLogger("TMSiActivity")
        if platform == "win32":
            tmsifolder = os.path.join(get_documents_path(), "TMSi","Activity")
            if not os.path.exists(tmsifolder):
                os.makedirs(tmsifolder)
            perf_handler = logging.FileHandler(os.path.join(tmsifolder, "__activity{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        else:
            perf_handler = logging.FileHandler("__activity{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(ACTIVITY_LOG)
        perf_handler.addFilter(LoggerFilter(ACTIVITY_LOG))
        self.__tmsi_perf.handlers = [perf_handler]

    def log(self, message):
        """Log the message with activity level.

        :param message: message to log.
        :type message: str
        """
        if self.activity_log_enabled:
            self.__tmsi_perf.log(level = ACTIVITY_LOG, msg = message)

class TMSiLoggerPerformance(metaclass = Singleton):
    """Class to handle the performance logs."""
    def __init__(self):
        """Initialize the performance logger."""
        self.performance_log_enabled = False
        if "TMSi_PERF" not in os.environ:
            return
        if os.environ["TMSi_PERF"] != "ON":
            return
        self.performance_log_enabled = True
        self.__tmsi_perf = logging.getLogger("TMSiPerformance")
        if platform == "win32":
            tmsifolder = os.path.join(get_documents_path(), "TMSi","Performances")
            if not os.path.exists(tmsifolder):
                os.makedirs(tmsifolder)
            perf_handler = logging.FileHandler(os.path.join(tmsifolder, "__performance.log"))
        else:
            perf_handler = logging.FileHandler("__performance.log")
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(PERFORMANCE_LOG)
        perf_handler.addFilter(LoggerFilter(PERFORMANCE_LOG))
        self.__tmsi_perf.handlers = [perf_handler]

    def log(self, message):
        """Log the message with performance level.

        :param message: message to log.
        :type message: str
        """
        if self.performance_log_enabled:
            self.__tmsi_perf.log(level = PERFORMANCE_LOG, msg = message)

