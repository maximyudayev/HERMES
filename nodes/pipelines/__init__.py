from nodes.pipelines.Pipeline import Pipeline

from nodes.pipelines.DummyPipeline import DummyPipeline
from nodes.pipelines.PytorchWorker import PytorchWorker

PIPELINES: dict[str, type[Pipeline]] = {
  "PytorchWorker": PytorchWorker,
  "DummyPipeline": DummyPipeline,
}
