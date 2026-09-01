"""
System Causal Graph Specification.

Builds NetworkX Directed Acyclic Graphs (DAGs) representing physical dependency
and domain causal mechanisms in Edge-IoT architectures:
    Workload -> CPU Utilization -> Latency
    Workload -> Memory Utilization -> Latency
    Network Utilization -> Packet Loss -> Latency
"""

import networkx as nx
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class SystemCausalGraph:
    """
    NetworkX Causal System Graph representing domain structure and causal assumptions.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_graph()

    def _build_default_graph(self):
        """
        Construct domain causal relationships:
            KNOWN_SIMULATION_CAUSALITY:
            Workload -> CPU Utilization -> Latency
            Workload -> Memory Utilization -> Latency
            Network Utilization -> Packet Loss -> Latency
        """
        edges = [
            ("workload", "cpu_utilization"),
            ("workload", "memory_utilization"),
            ("workload", "network_utilization"),
            ("cpu_utilization", "latency"),
            ("memory_utilization", "latency"),
            ("network_utilization", "packet_loss"),
            ("packet_loss", "latency"),
        ]
        self.graph.add_edges_from(edges)
        logger.info("SystemCausalGraph initialized with %d nodes and %d edges.",
                    self.graph.number_of_nodes(), self.graph.number_of_edges())

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_candidate_treatments(self) -> List[str]:
        """Return potential root-cause treatment variables in the DAG."""
        return [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
            "packet_loss",
            "latency",
        ]
