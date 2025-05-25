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

# ZeroMQ topics and message strings
TOPIC_KILL      = 'KILL'
CMD_HELLO       = 'HELLO'
CMD_ACK         = 'ACK'
CMD_START_TIME  = 'START_TIME'
CMD_GO          = 'GO'
CMD_END         = 'END'
CMD_EXIT        = 'EXIT?'
CMD_BYE         = 'BYE'
MSG_ON          = 'ON'
MSG_OFF         = 'OFF'
MSG_OK          = 'OK'

# Ports used for ZeroMQ by our system
PORT_BACKEND        = '42069'
PORT_FRONTEND       = '42070'
PORT_SYNC_HOST      = '42071'
PORT_SYNC_REMOTE    = '42072'
PORT_KILL           = '42066'
PORT_KILL_BTN       = '42065'
PORT_PAUSE          = '42067'

# Ports of connected devices/sensors
PORT_MOTICON      = '8888' # defined by the Moticon desktop app, putting data at the loopback address for listening
PORT_PROSTHESIS   = '51702' # defined by LabView code of VUB
PORT_GUI          = '8005'
PORT_EYE          = '50020'
PORT_VICON        = '801' # defined by Vicon software
PORT_MVN          = '9763' # defined by MVN Analyze

# IP addresses of devices on the network used by our system
DNS_LOCALHOST   = 'localhost'
IP_LOOPBACK     = '127.0.0.1'

IP_TERMINAL     = '192.168.0.100'
IP_PROSTHESIS   = '192.168.0.101'
IP_BACKPACK     = '192.168.0.103'
IP_VICON        = '192.168.0.104'
