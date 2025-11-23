from typing import Dict, List, Optional, Tuple, Union
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np


class FedNovaLite(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the previous global model to calculate updates (Delta)
        self.global_parameters: Optional[List[np.ndarray]] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Save the global model weights before sending to clients with numpy
        self.global_parameters = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Extract client updates and metadata
        updates = []
        tau_list = []
        num_examples_list = []

        # Calculate proper weights p_i = n_i/n (Section 2)
        total_examples = sum(fit_res.num_examples for _, fit_res in results) # n = sum of n_i
        weights = [fit_res.num_examples / total_examples for _, fit_res in results] # value of p_i

        for (client, fit_res), weight in zip(results, weights):
            # Get tau_i - total iterations of local solver per client
            tau_i = max(1, fit_res.metrics.get("tau_i", 1))  # Avoid division by zero
            tau_list.append(tau_i)
            num_examples_list.append(fit_res.num_examples)

            # Calculate client's parameter changes (Delta_i = x_i - x_global) (section 4.1)
            # (How much the client CHANGED the model)
            client_params = parameters_to_ndarrays(fit_res.parameters)

            delta_i = [c - g for g, c in zip(self.global_parameters, client_params)]

            # Normalize delta by local iterations/steps tau
            # "Normalized Gradient" = Total Change / Local Steps (equation 12)
            norm_grad = [layer / tau_i for layer in delta_i]
            updates.append(norm_grad)

        # Calculate effective steps (tau_eff) as weighted average of local steps (equation 12)
        tau_eff = sum(w * tau for w, tau in zip(weights, tau_list))
        # Aggregate normalized gradients with proper weighting p_i
        weighted_grads = [np.zeros_like(w) for w in self.global_parameters]

        for i, norm_grad in enumerate(updates):
            p_i = weights[i]
            for layer_idx, layer in enumerate(norm_grad):
                weighted_grads[layer_idx] += p_i * layer

        # Apply FedNova update: Global = Old Global - (tau_eff x aggregated_normalized_gradients)
        new_weights = [
            g - (tau_eff * wg) for g, wg in zip(self.global_parameters, weighted_grads)
        ]

        metrics = {"tau_eff": tau_eff, "server_round": server_round}
        return ndarrays_to_parameters(new_weights), metrics