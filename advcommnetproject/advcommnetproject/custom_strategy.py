from typing import Dict, List, Optional, Tuple
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import FitRes
import numpy as np


class FedNovaLite(FedAvg):
    """FedNova strategy implementing normalized averaging to handle client heterogeneity."""

    def __init__(self, fraction_train: float = 1.0, *args, **kwargs):
        super().__init__(
            fraction_train=fraction_train,
            # override the aggregation method used in FedAvg to implement w_i weight
            train_metrics_aggr_fn=self.fednova_metrics_aggregation, # see FedAvg documentation
            # https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html#flwr.serverapp.strategy.FedAvg
            *args,
            **kwargs
        )
        self.client_tau_history: Dict[str, List[int]] = {}  # Track tau_i per client (# of local steps)

    def fednova_metrics_aggregation(
            self, records: List[RecordDict], weighting_metric_name: str
    ) -> MetricRecord:
        """This code is to aggregate the FedNova metrics"""
        # Extract metrics from all records
        all_metrics = []
        tau_i_list = [] # collects tau metrics from client nodes

        for record in records:
            for record_item in record.metric_records.values():
                metrics_dict = dict(record_item)
                all_metrics.append(metrics_dict)
                tau_i = metrics_dict.get("tau_i", 1) # extract value of tau
                tau_i_list.append(tau_i) # build the list of all tau values

        # Use standard weighted aggregation to weigh contributions against dataset size
        aggregated_metrics = MetricRecord()

        # Get all metric keys
        metric_keys = set()
        for metrics in all_metrics:
            metric_keys.update(metrics.keys())

        # Remove the weighting key
        metric_keys.discard(weighting_metric_name)

        # Perform weighted averaging for each metric
        for key in metric_keys:
            weighted_sum = 0.0
            total_weight = 0.0

            for metrics in all_metrics:
                if key in metrics:
                    # calculate a client's weight
                    weight = metrics.get(weighting_metric_name, 1) # holds number of examples client trains on
                    weighted_sum += metrics[key] * weight
                    total_weight += weight

            # calculate the weighted average for each metric
            if total_weight > 0:
                aggregated_metrics[key] = weighted_sum / total_weight

        # Add FedNova-specific metrics
        if tau_i_list: # if we collected any tau values
            # Convert numpy types to native Python types
            aggregated_metrics["fednova_tau_avg"] = float(np.mean(tau_i_list)) # average local steps across clients
            aggregated_metrics["fednova_tau_std"] = float(np.std(tau_i_list)) # standard deviation
            aggregated_metrics["fednova_tau_min"] = int(np.min(tau_i_list)) # fewest local steps by a client
            aggregated_metrics["fednova_tau_max"] = int(np.max(tau_i_list)) # most local steps by a client

        return aggregated_metrics