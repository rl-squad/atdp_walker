import argparse
from train import train_stable, train_cleanrl

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="ATDP Walker PPO Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--lib", type=str, choices=["stable", "cleanrl"], required=True, help="Library to use")
    train_parser.add_argument("--job-id", type=str, required=True, help="Job ID for this run")
    train_parser.add_argument("--job-description", type=str, default="", help="Optional description of the training run")

    args = parser.parse_args()

    # Call the appropriate training function
    if args.command == "train":
        if args.lib == "stable":
            train_stable(args.job_id, args.job_description)
        elif args.lib == "cleanrl":
            train_cleanrl(args.job_id, args.job_description)

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
