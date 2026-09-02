# System Causal Graph Specification.
#
# Builds NetworkX Directed Acyclic Graphs (DAGs) representing physical dependency
# and domain causal mechanisms in Edge-IoT architectures:
#
# 1. Simulation Graph:
#     Workload -> CPU Utilization -> Latency
#     Workload -> Memory Utilization -> Latency
#     Network Utilization -> Packet Loss -> Latency
#
# 2. Sensor-Level Dataset Graphs (TON_IoT Domain Mechanisms):
#     Weather: Temperature -> Humidity, Temperature -> Pressure
#     Motion: Motion Status -> Light Status
#     Fridge: Fridge Temperature -> Temperature Condition

import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SystemCausalGraph:
    # NetworkX Causal System Graph representing domain structure and causal assumptions.
    # Supports both simulation system graphs and sensor dataset domain graphs.

    def __init__(self, mode: str = "simulation", device_type: Optional[str] = None):
        self.mode = mode
        self.device_type = device_type
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        if self.mode == "dataset":
            self._build_dataset_graph()
        else:
            self._build_simulation_graph()

    def _build_simulation_graph(self):
        # Construct simulation domain causal relationships:
        #     Workload -> CPU Utilization -> Latency
        #     Workload -> Memory Utilization -> Latency
        #     Network Utilization -> Packet Loss -> Latency
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
        logger.info("SystemCausalGraph (Simulation) initialized with %d nodes and %d edges.",
                    self.graph.number_of_nodes(), self.graph.number_of_edges())

    def _build_dataset_graph(self):
        # Construct domain-defined causal relationships for TON_IoT sensor telemetry.
        edges = []
        dev = str(self.device_type or "").lower()

        if "weather" in dev:
            edges = [
                ("temperature", "humidity"),
                ("temperature", "pressure"),
            ]
        elif "motion" in dev:
            edges = [
                ("motion_status", "light_status_on"),
            ]
        elif "fridge" in dev:
            edges = [
                ("fridge_temperature", "temp_condition_high"),
            ]
        elif "modbus" in dev:
            edges = [
                ("FC1_Read_Input_Register", "FC3_Read_Holding_Register"),
                ("FC2_Read_Discrete_Value", "FC4_Read_Coil"),
            ]
        elif "gps" in dev:
            edges = [
                ("latitude", "longitude"),
            ]
        elif "garage" in dev:
            edges = [
                ("door_state_open", "sphone_signal_true"),
            ]

        self.graph.add_edges_from(edges)
        logger.info(f"SystemCausalGraph (Dataset: {self.device_type}) initialized with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_candidate_treatments(self) -> List[str]:
        # Return potential root-cause treatment variables in the DAG.
        if self.mode == "dataset":
            return list(self.graph.nodes())
        return [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
            "packet_loss",
            "latency",
        ]

