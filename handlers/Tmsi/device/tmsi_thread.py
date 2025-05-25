'''
(c) 2022 Twente Medical Systems International B.V., Oldenzaal The Netherlands

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
 * @file sampling_thread.py 
 * @brief 
 * Sampling Thread interface
 */


'''

import threading
import time


class TMSiThread(threading.Thread):
    """A class to handle all the sampling threads."""
    def __init__(self, looping_function, pause = 0.01, name = "TMSi Thread"):
        """_summary_

        :param looping_function: the function which must be exectuted.
        :type looping_function: function
        :param pause: pause time between each loop of the thread, defaults to 0.01
        :type pause: float, optional
        :param name: name of the thread, defaults to "Sampling Thread"
        :type name: str, optional
        """
        super().__init__()
        self.name = name
        self._looping_function = looping_function
        self._pause = pause

    def get_pause(self):
        """Get the pause time of the thread.

        :return: the pause time between each loop of the thread.
        :rtype: float
        """
        return self._pause
    
    def run(self):
        """Run the thread.
        """
        self._looping = True
        while self._looping:
            self._looping_function()
            time.sleep(self._pause)

    def set_pause(self, pause):
        """Set the pause time of the thread.

        :param pause: the pause time between each loop of the thread.
        :type pause: float
        """
        self._pause = pause

    def stop(self):
        """Stop the thread.
        """
        self._looping = False