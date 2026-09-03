from dataclasses import dataclass
from typing import Any

from graphviz import Digraph


@dataclass
class TensorNode:
    name: str
    operation: str
    shape: tuple
    data: Any = None
    grad: Any = None


class TensorGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(
        self,
        name,
        operation,
        tensor,
    ):
        node = TensorNode(
            name=name,
            operation=operation,
            shape=tuple(tensor.shape),
            data=tensor.detach().cpu(),
            grad=None,
        )

        self.nodes.append(node)

        return node

    def add_edge(self, source, target):
        self.edges.append((source.name, target.name))

    def draw(self, filename="tensor_graph"):
        dot = Digraph(
            format="png",
            graph_attr={
                "rankdir": "LR",
                "splines": "ortho",
                "nodesep": "0.5",
                "ranksep": "0.8",
            },
        )

        for node in self.nodes:
            data = self._format_data(node.data)

            label = (
                f"{{"
                f"{node.name}"
                f"|"
                f"op: {node.operation}"
                f"|"
                f"shape: {node.shape}"
                f"|"
                f"data: {data}"
                f"}}"
            )

            dot.node(
                node.name,
                label=label,
                shape="record",
            )

        for source, target in self.edges:
            dot.edge(source, target)

        output_path = dot.render(
            filename,
            cleanup=True,
        )

        return output_path

    @staticmethod
    def _format_data(data):
        if data is None:
            return "None"

        flat = data.flatten()

        if flat.numel() > 8:
            values = flat[:8].tolist()
            return str(values) + " ..."

        return str(flat.tolist())