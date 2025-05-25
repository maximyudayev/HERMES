from nodes.producers.Producer import Producer


PRODUCERS: dict[str, type[Producer]] = {}
try:
  from nodes.producers.DotsStreamer import DotsStreamer
  PRODUCERS["DotsStreamer"] = DotsStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"DotsStreamer.", flush=True)

try:
  from nodes.producers.CameraStreamer import CameraStreamer
  PRODUCERS["CameraStreamer"] = CameraStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"CameraStreamer.", flush=True)

try:
  from nodes.producers.CometaStreamer import CometaStreamer
  PRODUCERS["CometaStreamer"] = CometaStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"CometaStreamer.", flush=True)

try:
  from nodes.producers.EyeStreamer import EyeStreamer
  PRODUCERS["EyeStreamer"] = EyeStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"EyeStreamer.", flush=True)

try:
  from nodes.producers.CyberlegStreamer import CyberlegStreamer
  PRODUCERS["CyberlegStreamer"] = CyberlegStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"CyberlegStreamer.", flush=True)

try:
  from nodes.producers.InsoleStreamer import InsoleStreamer
  PRODUCERS["InsoleStreamer"] = InsoleStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"InsoleStreamer.", flush=True)

try:
  from nodes.producers.MvnAnalyzeStreamer import MvnAnalyzeStreamer
  PRODUCERS["MvnAnalyzeStreamer"] = MvnAnalyzeStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"MvnAnalyzeStreamer.", flush=True)

try:
  from nodes.producers.ExperimentControlStreamer import ExperimentControlStreamer
  PRODUCERS["ExperimentControlStreamer"] = ExperimentControlStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"ExperimentControlStreamer.", flush=True)

try:
  from nodes.producers.AwindaStreamer import AwindaStreamer
  PRODUCERS["AwindaStreamer"] = AwindaStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"AwindaStreamer.", flush=True)

try:
  from nodes.producers.MoxyStreamer import MoxyStreamer
  PRODUCERS["MoxyStreamer"] = MoxyStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"MoxyStreamer.", flush=True)

try:
  from nodes.producers.TmsiStreamer import TmsiStreamer
  PRODUCERS["TmsiStreamer"] = TmsiStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"TmsiStreamer.", flush=True)

try:
  from nodes.producers.ViconStreamer import ViconStreamer
  PRODUCERS["ViconStreamer"] = ViconStreamer
except ImportError as e:
  print(e, "\nSkipping %s"%"ViconStreamer.", flush=True)

try:
  from nodes.producers.DummyProducer import DummyProducer
  PRODUCERS["DummyProducer"] = DummyProducer
except ImportError as e:
  print(e, "\nSkipping %s"%"DummyProducer.", flush=True)
