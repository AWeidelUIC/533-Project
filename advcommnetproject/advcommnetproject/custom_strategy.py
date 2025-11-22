from typing import Dict, List, Optional, Tuple, Union
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np


class FedNovaLite(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We need to track the previous global model to calculate updates (Delta)
        self.global_parameters: Optional[List[np.ndarray]] = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ):
        # Save the global model weights before sending to clients
        self.global_parameters = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # 1. Extract Weights and Tau from results
        # We need to reconstruct the 'update' (Delta) for each client
        updates = []
        tau_list = []
        num_examples_list = []

        for client, fit_res in results:
            # Get tau from metrics (default to 1 to avoid div/0)
            tau_i = fit_res.metrics.get("tau_i", 1)
            tau_list.append(tau_i)
            num_examples_list.append(fit_res.num_examples)

            # Get client parameters
            client_params = parameters_to_ndarrays(fit_res.parameters)

            # Calculate Delta_i = Global - Client
            # (How much the client CHANGED the model)
            delta_i = [
                g - c for g, c in zip(self.global_parameters, client_params)
            ]

            # Normalize Delta by tau_i (FedNova core logic)
            # "Normalized Gradient" = Total Change / Local Steps
            norm_grad = [layer / tau_i for layer in delta_i]
            updates.append(norm_grad)

        # 2. Calculate Effective Tau (tau_eff)
        # This is usually the weighted average of local steps
        total_examples = sum(num_examples_list)
        tau_eff = sum([t * (n/total_examples) for t, n in zip(tau_list, num_examples_list)])

        # 3. Aggregate Normalized Gradients
        # aggregated_update = Sum( p_i * norm_grad_i )
        weighted_grads = [np.zeros_like(w) for w in self.global_parameters]

        for i, norm_grad in enumerate(updates):
            p_i = num_examples_list[i] / total_examples
            for layer_idx, layer in enumerate(norm_grad):
                weighted_grads[layer_idx] += p_i * layer

        # 4. Apply to Global Model
        # New Global = Old Global - (tau_eff * aggregated_update)
        # (Treating server learning rate as 1.0 for simplicity)
        new_weights = [
            g - (tau_eff * wg)
            for g, wg in zip(self.global_parameters, weighted_grads)
        ]

        # 5. Return parameters and custom metrics (optional)
        metrics = {"tau_eff": tau_eff}
        return ndarrays_to_parameters(new_weights), metrics 
