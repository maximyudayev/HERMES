from nodes.pipelines.Pipeline import Pipeline

from nodes.pipelines.DummyPipeline import DummyPipeline

PIPELINES: dict[str, type[Pipeline]] = {
  "DummyPipeline": DummyPipeline,
}

try:
  from nodes.pipelines.PytorchWorker import PytorchWorker
  PIPELINES["PytorchWorker"] = PytorchWorker
except ImportError as e:
  print(e, "\nSkipping %s"%"PytorchWorker.", flush=True)
