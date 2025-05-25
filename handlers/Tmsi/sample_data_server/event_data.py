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
 * @file event_data.py 
 * @brief 
 * Event data structure.
 */


'''

class EventData:
    """Class to handle the event data.
    """
    def __init__(self, event: "TMSiEvent"):
        """Initialize the Event data.

        :param event: event
        :type event: TMSiEvent
        """
        self.event = event

class EventDataConsumer:
    """Class to handle the event data consumers.
    """
    def __init__(self, id, q):
        """Initialize the event data consumer.

        :param id: id of the server to consume.
        :type id: int
        :param q: consumer.
        :type q: Queue
        """
        self.id = id
        self.q = q