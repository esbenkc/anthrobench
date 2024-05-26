import pandas as pd
from utils import eval_responses, save_result, request_response


def run(dataset="anthrobench.csv", model="gpt-4o"):
    # Load dataset
    df = pd.read_csv(dataset)
    # Run model over dataset
    df["response"] = df["question"].apply(lambda x: request_response(x, model))
    # Evaluate responses
    responses = eval_responses(df)
    # Convert returned list to dataframe
    df["evaluation"] = responses
    # Save results
    save_result(df, f"results/anthrobench_results_{model}.csv")
