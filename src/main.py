from anthrobench import run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ArcHydro schema")
    parser.add_argument(
        "--dataset", type=str, default="anthrobench.tsv", help="Dataset to use"
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    args = parser.parse_args()
    run(args.dataset, args.model)
