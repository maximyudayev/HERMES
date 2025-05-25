from nodes.consumers.Consumer import Consumer

from nodes.consumers.DataLogger import DataLogger
from nodes.consumers.DataVisualizer import DataVisualizer
from nodes.consumers.DummyConsumer import DummyConsumer

CONSUMERS: dict[str, type[Consumer]] = {
  "DataLogger": DataLogger,
  "DataVisualizer": DataVisualizer,
  "DummyConsumer": DummyConsumer,
}
