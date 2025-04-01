import argparse
from train import StableTrainer, CleanRLTrainer, RLlibTrainer, GarageTrainer, CustomTrainer



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="ATDP Walker PPO Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--lib", type=str,
                              choices=["stable", "cleanrl", "rllib", "garage", "custom", "hf_sb3"],
                              required=True, help="Library to use")
    train_parser.add_argument("--job-id", type=str, required=True, help="Job ID for this run")
    train_parser.add_argument("--job-description", type=str, default="", help="Optional description of the training run")

    args = parser.parse_args()

    # Map string to the correct trainer class
    trainer_map = {
        "stable": StableTrainer,
        "cleanrl": CleanRLTrainer,
        "rllib": RLlibTrainer,
        "garage": GarageTrainer,
        "custom": CustomTrainer,
    }

    # Instantiate and run the selected trainer
    if args.command == "train":
        trainer_class = trainer_map.get(args.lib)
        trainer = trainer_class(args.job_id, args.job_description)
        trainer.train()


if __name__ == "__main__":
    main()
