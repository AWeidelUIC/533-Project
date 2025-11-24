"""AdvCommNetProject: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from advcommnetproject.task import Net
from advcommnetproject.task import central_evaluate

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    proximal_mu: float = context.run_config["proximal-mu"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedProx with proximal_mu
    strategy = FedProx(proximal_mu=proximal_mu)

    # Start strategy, run FedProx for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "proximal_mu": proximal_mu
        }),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")