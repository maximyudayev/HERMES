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
 * @file decorators.py 
 * @brief 
 * Decorator class to handle decorative functions.
 */


'''

import time
import os
from functools import wraps

from .tmsi_logger import TMSiLoggerPerformance
from .tmsi_logger import TMSiLogger


def LogPerformances(func):
    @wraps(func)
    def performance_logger(*args, **kwargs):
        if "TMSi_ENV" in os.environ:
            env = os.environ["TMSi_ENV"]
        else:
            env = "DLL"
        if "TMSi_PERF" not in os.environ:
            response = func(*args, **kwargs)
        elif os.environ["TMSi_PERF"] == "ON":
            tic = time.perf_counter()
            response = func(*args, **kwargs)
            toc = time.perf_counter()
            TMSiLoggerPerformance().log("{} | {}: {:.3f} ms".format(env, func.__qualname__, (toc - tic)*1_000))
        else:
            response = func(*args, **kwargs)
        return response
    return performance_logger

def Retry(n_retry : int):
    def retry_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(n_retry):
                try:
                    return func(*args, **kwargs)
                except Exception as exec:
                    last_exception = exec
                    TMSiLogger().warning("Retry: {} - {}".format(func.__qualname__, exec))
                time.sleep(0.5)
            if last_exception:
                raise last_exception
            else:
                raise ValueError("{} failed but no exception was raised.".format(func.__qualname__))
        return wrapper
    return retry_decorator