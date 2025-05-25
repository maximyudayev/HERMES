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
 * @file sample_data.py 
 * @brief 
 * Sample data structure.
 */


'''

class SampleSet:
    """Class to handle sample sets.
    """
    def __init__(self, num_samples, samples):
        """Initialize sample set.

        :param num_samples: number of samples
        :type num_samples: int
        :param samples: samples
        :type samples: list[sample]
        """

        self.num_samples = num_samples
        self.samples = samples

class SampleData:
    """Class to handle sample data.
    """
    def __init__(self, num_sample_sets, num_samples_per_sample_set, samples):
        """Initialize the sample data.

        :param num_sample_sets: number of sets of sample
        :type num_sample_sets: int
        :param num_samples_per_sample_set: number of samples in each set.
        :type num_samples_per_sample_set: int
        :param samples: samples
        :type samples: list[sample]
        """
        self.num_sample_sets = num_sample_sets
        self.num_samples_per_sample_set = num_samples_per_sample_set
        self.samples = samples

class SampleDataConsumer:
    """Class to handle the sample data consumers.
    """
    def __init__(self, id, q):
        """Initialize the sample data consumer.

        :param id: id of the server to consume.
        :type id: int
        :param q: consumer.
        :type q: Queue
        """
        self.id = id
        self.q = q