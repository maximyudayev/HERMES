############
#
# Copyright (c) 2024 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

import threading
from nodes.Broker import Broker
from utils.mp_utils import launch_callable
from utils.time_utils import *

import os
import sys
import yaml


if __name__ == '__main__':
  # Parse YAML config file.
  # $> python ./main.py configs/example/template.yml
  # TODO: do all the configuration checking to avoid conflicting or missing parameters.
  config_path: str = sys.argv[1]
  with open(config_path, "r") as f:
    try:
      config: dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
      print(e)

  with open(config['logging_spec']['video_codec_config_filepath'], "r") as f:
    try:
      config['logging_spec']['video_codec'] = yaml.safe_load(f)
    except yaml.YAMLError as e:
      print(e)


  # Initialize folders and other chore data, and share programmatically across Node specs. 
  script_dir: str = os.path.dirname(os.path.realpath(__file__))
  (log_time_str, log_time_s) = get_time_str(return_time_s=True)
  log_dir_root: str = os.path.join(script_dir, 
                                   'data',
                                   config['project'],
                                   config['trial_type'],
                                   '{0}_S{1}_{2}'.format(get_time_str(format='%Y-%m-%d'), 
                                                         str(config['subject_id']).zfill(3), 
                                                         str(config['trial_id']).zfill(2)))
  log_dir: str = os.path.join(log_dir_root, log_time_str)
  # Initialize a file for writing the log history of all printouts/messages.
  log_history_filepath: str = os.path.join(log_dir, '%s_log_history.txt' % (log_time_str))

  # TODO: ssh into remote IPs, distribute the log_dir and launch main.py on each device.
  # TODO: distribute log_history.txt across Nodes and prepend each with Node tag.

  os.makedirs(log_dir, exist_ok=True)

  config['logging_spec']['log_dir'] = log_dir

  # Add logging spec to each producer.
  for spec in config['producer_specs']:
    spec['logging_spec'] = config['logging_spec']

  # Add logging spec to each consumer.
  for spec in config['consumer_specs']:
    spec['logging_spec']['log_dir'] = log_dir
    spec['log_history_filepath'] = log_history_filepath

  producer_specs: list[dict] = config['producer_specs']
  consumer_specs: list[dict] = config['consumer_specs']
  pipeline_specs: list[dict] = config['pipeline_specs']


  # Create the broker and manage all the components of the experiment.
  local_broker: Broker = Broker(host_ip=config['host_ip'],
                                node_specs=producer_specs+consumer_specs+pipeline_specs,
                                is_master_broker=config['is_master_broker'],
                                print_status=config['print_status'], 
                                print_debug=config['print_debug'])

  # Connect broker to remote publishers at the wearable PC to get data from the wearable sensors.
  for ip in config['remote_publisher_ips']:
    local_broker.connect_to_remote_broker(addr=ip)

  # Expose local wearable data to remote subscribers (e.g. lab PC in AidFOG project).
  if config['remote_subscriber_ips']:
    local_broker.expose_to_remote_broker(config['remote_subscriber_ips'])
  
  # Subscribe to the KILL signal of a remote machine.
  if config['is_remote_kill']:
    local_broker.subscribe_to_killsig(addr=config['remote_kill_ip'])

  is_quit = False
  # Run broker's main until user exits in GUI or 'q' in terminal.
  t = threading.Thread(target=launch_callable, args=(local_broker, config['duration_s']))
  t.start()
  while not is_quit:
    is_quit = input("Enter 'q' to exit: ") == 'q'
  local_broker.set_is_quit()
  t.join()

  # TODO: collect files from remote IPs
