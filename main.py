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
from utils.argparse_utils import *
from utils.user_input_utils import *

import os
import yaml
import argparse
from fabric import Connection


__version = 'v0.1.0'

# global params for testing remote connection
LAPTOP_IP = "192.168.137.34"
LAPTOP_USER = "u0170757"
LAPTOP_KEY = "C:/Users/Owner/.ssh/nuc_rsa"  
REMOTE_PROJECT_PATH = "C:/Users/u0170757/OneDrive - KU Leuven/Documents/AID-FOG/data_acquisition"
REMOTE_MAIN = "main.py"

if __name__ == '__main__':

  parser = argparse.ArgumentParser(prog='HERMES',
                                   description='Heterogeneous edge realtime measurement and execution system '
                                               'for continual multimodal data acquisition and processing.',
                                   epilog='Copyright (c) 2024 Maxim Yudayev and KU Leuven eMedia Lab.\n'
                                          'Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo '
                                          'projects of prof. Bart Vanrumste, by Maxim Yudayev [https://yudayev.com].',
                                  formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--verbose', '-v',
                      action='count',
                      default=0,
                      help='increase level of logging verbosity [0,3]')
  parser.add_argument('--version',
                      action='version',
                      version='%(prog)s ' + __version)

  parser.add_argument('--experiment',
                      nargs='*',
                      action=ParseExperimentKwargs,
                      help='key-value pair tags detailing the experiment, used for '
                           'directory creation and metadata on files')
  parser.add_argument('--time', '-t',
                      type=float,
                      dest='log_time_s',
                      default=get_time(),
                      help='master start time of the system')
  parser.add_argument('--duration', '-d',
                      type=int,
                      dest='duration_s',
                      default=None,
                      help='duration in seconds, if using for recording only (to be used only by master)')
  parser.add_argument('--config_file',
                      type=validate_path,
                      default=None,
                      help='path to the configuration file for the current host device, '
                           'instead of the CLI arguments')

  # parser.add_argument('--host_ip',
  #                     type=validate_ip,
  #                     help='LAN IPv4 address of the host device')
  # parser.add_argument('--is_master',
  #                     dest='is_master_broker',
  #                     action='store_true',
  #                     help='flag to set host as Master Broker that others connect to')
  # parser.add_argument('--subscriber_ips',
  #                     nargs='*',
  #                     dest='remote_subscriber_ips',
  #                     type=validate_ip,
  #                     default=list(),
  #                     help='list of IPv4 devices listening to data of the host')
  # parser.add_argument('--publisher_ips',
  #                     nargs='*',
  #                     dest='remote_publisher_ips',
  #                     type=validate_ip,
  #                     default=list(),
  #                     help='list of IPv4 devices to listen to for their data')
  # parser.add_argument('--kill',
  #                     dest='is_remote_kill',
  #                     action='store_true',
  #                     help='flag to set host to listen to remote KILLSIG (only for slave devices)')
  # parser.add_argument('--kill_ip',
  #                     dest='remote_kill_ip',
  #                     type=validate_ip,
  #                     help='LAN IPv4 address of device delegating KILLSIG (only for slave devices)')


  # parser.add_argument('--logging_spec',
  #                     nargs='*',
  #                     action=ParseLoggingKwargs,
  #                     help='key-value pair tags configuring logging modules of each Node')

  # parser.add_argument('--producer_specs',
  #                     nargs='*',
  #                     action=ParseNodeKwargs,
  #                     default=list(),
  #                     help='key-value pair tags detailing local producer Nodes of the host')
  # parser.add_argument('--consumer_specs',
  #                     nargs='*',
  #                     action=ParseNodeKwargs,
  #                     default=list(),
  #                     help='key-value pair tags detailing local consumer Nodes of the host')
  # parser.add_argument('--pipeline_specs',
  #                     nargs='*',
  #                     action=ParseNodeKwargs,
  #                     default=list(),
  #                     help='key-value pair tags detailing local pipeline Nodes of the host')


  # Parse launch arguments.
  args = parser.parse_args()
  # Create a GUI to ask for user input about the experiment information and file naming
  user_input_gui = ExperimentGUI(args)
  user_args = user_input_gui.get_experiment_inputs()
  # Override CLI arguments with a config file.
  if user_args.config_file is not None:
    with open(user_args.config_file, "r") as f:
      try:
        config: dict = yaml.safe_load(f)
        parser.set_defaults(**config)
      except yaml.YAMLError as e:
        print(e)
        exit('Error parsing CLI inputs.')
    args = parser.parse_args()
    # Overwrite args with the user selected ones in the GUI
    for k, v in vars(user_args).items():
      if hasattr(args, k):
          setattr(args, k, v) 

  # Load video codec spec.
  try: 
    if 'stream_video' in args.logging_spec and args.logging_spec['stream_video']:
      with open(args.logging_spec['video_codec_config_filepath'], "r") as f:
        try:
          args.logging_spec['video_codec'] = yaml.safe_load(f)
        except yaml.YAMLError as e:
          print(e)
  except Exception:
    pass

  # Initialize folders and other chore data, and share programmatically across Node specs.
  script_dir: str = os.path.dirname(os.path.realpath(__file__))
  (log_time_str, log_time_s) = get_time_str(time_s=args.log_time_s, return_time_s=True)
  log_dir: str = os.path.join(script_dir,
                              'data',
                               f'project_{args.experiment["project"]}', # to be removed 
                               f'subject_{args.experiment["subject"]}_{args.experiment["group"]}',
                               f'session_{args.experiment["session"]}')
                              #*map(lambda tup: '_'.join(tup), args.experiment.items()))

  # Initialize a file for writing the log history of all printouts/messages.
  log_history_filepath: str = os.path.join(log_dir, '%s.log'%args.host_ip)

  # ssh into remote IPs, distribute Node configs, experiment settings, and launch main.py on each device.
  # exp_args = ' '.join([f'{k}={v}' for k, v in args.experiment.items()])
  # cmd_args = f"--experiment {exp_args} --config_file {args.config_file}"

  # # Construct connection and run
  # conn = Connection(
  #     host=LAPTOP_IP,
  #     user=LAPTOP_USER,
  #     connect_kwargs={"key_filename": LAPTOP_KEY}
  # )
  # conn.run(f'cd "{REMOTE_PROJECT_PATH}" && python {REMOTE_MAIN} {cmd_args}', hide=False)

  args.logging_spec['log_dir'] = log_dir
  args.logging_spec['experiment'] = args.experiment
  args.logging_spec['log_time_s'] = log_time_s

  # Add logging spec to each producer.
  for spec in args.producer_specs:
    spec['logging_spec'] = args.logging_spec

  # Add logging spec to each consumer.
  for spec in args.consumer_specs:
    spec['logging_spec']['log_dir'] = log_dir
    spec['logging_spec']['experiment'] = args.experiment # type: ignore
    spec['logging_spec']['log_time_s'] = log_time_s
    spec['log_history_filepath'] = log_history_filepath

  # Add logging spec to each consumer.
  for spec in args.pipeline_specs:
    spec['logging_spec']['log_dir'] = log_dir
    spec['logging_spec']['experiment'] = args.experiment # type: ignore
    spec['logging_spec']['log_time_s'] = log_time_s
    spec['log_history_filepath'] = log_history_filepath

  producer_specs: list[dict] = args.producer_specs
  consumer_specs: list[dict] = args.consumer_specs
  pipeline_specs: list[dict] = args.pipeline_specs

  # Create the broker and manage all the components of the experiment.
  local_broker: Broker = Broker(host_ip=args.host_ip,
                                node_specs=producer_specs+consumer_specs+pipeline_specs,
                                is_master_broker=args.is_master_broker)

  # Connect broker to remote publishers at the wearable PC to get data from the wearable sensors.
  for ip in args.remote_publisher_ips:
    local_broker.connect_to_remote_broker(addr=ip)

  # Expose local wearable data to remote subscribers (e.g. lab PC in AidFOG project).
  if args.remote_subscriber_ips:
    local_broker.expose_to_remote_broker(args.remote_subscriber_ips)
  
  # Subscribe to the KILL signal of a remote machine.
  if args.is_remote_kill:
    local_broker.subscribe_to_killsig(addr=args.remote_kill_ip)

  # Only the master broker can terminate the experiment via the terminal command.
  if args.is_master_broker:
    is_quit = False
    # Run broker's main until user exits in GUI or 'q' in terminal.
    t = threading.Thread(target=launch_callable, args=(local_broker, args.duration_s))
    t.start()
    while not is_quit:
      is_quit = input("Enter 'q' to exit: ") == 'q'
    local_broker.set_is_quit()
    t.join()
  else:
    local_broker()  

  # collect files from remote IPs at the end of the experiments.
  # conn.get(f"{REMOTE_PROJECT_PATH}/data/*.hdf5", local=log_dir)
  # conn.get(f"{REMOTE_PROJECT_PATH}/data/*.mkv", local=log_dir)
  # cleanup remote files
  # conn.run(f'rm {REMOTE_PROJECT_PATH}/data/*.hdf5')
  # conn.run(f'rm {REMOTE_PROJECT_PATH}/data/*.mkv')
