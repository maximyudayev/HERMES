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

import argparse
from typing import Any, Sequence


def validate_ip(s: str) -> str:
  try:
    a = s.split('.')
    assert len(a) == 4
    for x in a:
      assert x.isdigit()
      n = int(x)
      assert n >= 0 and n <= 255
    return s
  except:
    raise argparse.ArgumentTypeError("Not a valid IPv4 address: ", s)


def validate_path(s: str) -> str:
  try:
    return s
  except:
    raise argparse.ArgumentTypeError("Invalid path to config file: ", s)
    # raise argparse.ArgumentTypeError("Config file does not exist: ", s)


def parse_type(s: str) -> int | float | str:
  if s.isdigit(): return int(s)
  elif s == 'True':
    return True
  elif s == 'False':
    return False
  else:
    try:
      return float(s)
    except:
      return s


class ParseExperimentKwargs(argparse.Action):
  def __call__(self, parser, namespace, values, option_string = None):
    if isinstance(values, (list, tuple)):
      setattr(namespace, self.dest, dict())
      for value in values:
        key, value = value.split('=')
        getattr(namespace, self.dest)[key] = value


class ParseLoggingKwargs(argparse.Action):
  def __call__(self, parser, namespace, values, option_string = None):
    if isinstance(values, (list, tuple)):
      setattr(namespace, self.dest, dict())
      for value in values:
        if '=' in value:
          key, val = value.split('=')
          getattr(namespace, self.dest)[key] = parse_type(val)
        else:
          getattr(namespace, self.dest)[value] = True


class ParseNodeKwargs(argparse.Action):
  def __call__(self, parser, namespace, values, option_string = None):
    if isinstance(values, (list, tuple)):
      new_items = list()
      # Parse the input values as a dictionary
      id = -1
      for value in values:
        key, val = value.split('=')
        if key == 'class':
          id += 1
          new_items.append(dict([(key, val)]))
        elif ';' in val:
          new_items[id][key] = dict()
          for pair_str in val.split(';'):
            k, v = pair_str.split(':')
            new_items[id][key][k] = v
        elif ',' in val:
          new_items[id][key] = list(map(parse_type, val.split(',')))
        else:
          new_items[id][key] = parse_type(val)
      # Extend the list with the new dictionary
      items = getattr(namespace, self.dest, list())
      items.extend(new_items)
      setattr(namespace, self.dest, items)
