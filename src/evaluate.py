import os
import json
import typer
import pandas as pd
from typing import List, Optional
from pathlib import Path
from metrics import (
    jaccard_index_dictionary,
    jaccard_index_scispacy,
    cosine_similarity_biobert,
    machine_translation_metrics,
    bert_score_metric,
)


# Create the Typer app
app = typer.Typer()

# Define available metrics
AVAILABLE_METRICS = [
    "jaccard_index_dictionary",
    # 'jaccard_index_scispacy',
    "cosine_similarity_biobert",
    "machine_translation_metrics",
    "bert_score_metric",
]


def save_data(dir_path: str, results: dict, mean_results: dict) -> None:
    """
    Save computed metric results to JSON files.

    Args:
        dir_path (str): The directory where results should be saved.
        results (dict): The computed metric results.
        mean_results (dict): The averaged metric results.

    Returns:
        None
    """
    # Create the directory if it does not exist
    output_dir = Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"
    mean_output_file = output_dir / "results_averaged.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    with open(mean_output_file, "w") as f:
        json.dump(mean_results, f, indent=4)

    print(f"Finished! Saved results per text pair at: {output_file} and averaged results at: {mean_output_file}")

def load_data(input_file: str) -> tuple:
    """
    Load input data from a CSV or JSON file.

    Args:
        input_file (str): Path to the input file containing 'generated' and 'original' texts.

    Returns:
        tuple: Two lists - (generated_texts, original_texts).

    Raises:
        ValueError: If the file type is unsupported or missing required columns.
    """
    # Check file extension
    _, file_extension = os.path.splitext(input_file)
    file_extension = file_extension.lower()

    if file_extension == ".csv":
        # Load CSV file
        data = pd.read_csv(input_file)
    elif file_extension == ".json":
        # Load JSON file
        with open(input_file, "r") as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file type. Only .csv and .json are allowed.")

    data.columns = data.columns.str.lower()
    
    # Check if 'generated' and 'original' columns exist
    if not {"generated", "original"}.issubset(data.columns):
        raise ValueError("The file must contain 'generated' and 'original' columns or keys.")

    # Extract the 'generated' and 'original' columns as lists
    generated = data["generated"].tolist()
    original = data["original"].tolist()

    return generated, original


def run_evaluation(original_texts: List[str], generated_texts: List[str], metrics: Optional[List[str]] = None, dict_path: Optional[str] = None) -> tuple:
    """
    Compute various machine translation and text similarity metrics.

    Args:
        original_texts (List[str]): List of reference texts (ground truth).
        generated_texts (List[str]): List of generated texts (hypotheses).
        metrics (Optional[List[str]]): List of metrics to compute. Defaults to all available metrics.
        dict_path (Optional[str]): Path to the dictionary file for custom keyword-based Jaccard index.

    Returns:
        tuple: 
            - results (dict): Dictionary with per-instance metric results.
            - averaged_results (dict): Dictionary with mean scores for each metric.
    """
    results = {}
    metrics = AVAILABLE_METRICS if metrics is None else metrics
    dict_path = "./data/dictionary.json" if dict_path is None else dict_path

    for metric in metrics:
        typer.echo(f"Computing {metric}...")
        
        if metric == "jaccard_index_dictionary":
            jacc_dict_score = jaccard_index_dictionary(original_texts, generated_texts, dict_path)
            results["jaccard_index_dictionary"] = jacc_dict_score

        elif metric == "jaccard_index_scispacy":
            jacc_scispacy_score = jaccard_index_scispacy(original_texts, generated_texts)
            results["jacc_scispacy_score"] = jacc_scispacy_score

        elif metric == "cosine_similarity_biobert":
            cosine_sim = cosine_similarity_biobert(original_texts, generated_texts)
            results["cosine_similarity_biobert"] = cosine_sim

        elif metric == "machine_translation_metrics":
            scores = machine_translation_metrics(original_texts, generated_texts)

            # Store per-report results as lists
            results["BLEU-1"] = scores["BLEU-1"]
            results["BLEU-2"] = scores["BLEU-2"]
            results["BLEU-3"] = scores["BLEU-3"]
            results["BLEU-4"] = scores["BLEU-4"]
            results["ROUGE-L"] = scores["ROUGE-L"]
            results["METEOR"] = scores["METEOR"]
            
            # CIDEr is computed on the whole dataset, so it’s a single number
            results["CIDEr"] = scores["CIDEr"]

        elif metric == "bert_score_metric":
            p, r, f1 = bert_score_metric(original_texts, generated_texts)
            results["Bertscore_p"] = p
            results["Bertscore_r"] = r
            results["Bertscore_f1"] = f1

        else:
            typer.echo(f"Metric {metric} not available. Skipping...")

    averaged_results = compute_mean_results(results)
    return results, averaged_results


def compute_mean_results(results: dict) -> dict:
    """
    Compute mean values for metrics that have per-instance scores.

    Args:
        results (dict): Dictionary with metric scores (some may be lists).

    Returns:
        dict: Dictionary containing the mean of per-instance scores.
    """
    results_averaged = {}
    for k, v in results.items():
        # Calculate the mean of lists; keep single values as-is
        results_averaged[f"{k}_mean"] = (sum(v) / len(v)) if isinstance(v, list) and v else v

    return results_averaged


@app.command()
def main(
    input_file: str = typer.Argument(
        ..., help="Path to the input file with the original and generated texts"
    ),
    output_dir: str = typer.Argument(
        ..., help="Path where the file with the computed metrics will be stored"
    ),
    dict_path: Optional[str] = typer.Option(
        None, "--dict-path", "-d", help="Path to dictionary file with custom keywords"
    ),
    metrics: Optional[List[str]] = typer.Option(
        None,
        "--metrics",
        "-m",
        help=f"List of metrics to compute. Available options: {', '.join(AVAILABLE_METRICS)}. Defaults to all metrics.",
    ),
) -> None:
    """
    Main function to load data, run evaluations, and save results.

    Args:
        input_file (str): Path to the input CSV/JSON file containing 'original' and 'generated' texts.
        output_dir (str): Path to store computed metric results.
        dict_path (Optional[str]): Path to a dictionary file for Jaccard similarity calculations.
        metrics (Optional[List[str]]): List of selected metrics. Defaults to all available metrics.

    Returns:
        None
    """
    original_texts, generated_texts = load_data(input_file)
    results_dict, mean_results_dict = run_evaluation(
        original_texts,
        generated_texts,
        metrics if metrics else None,
        dict_path if dict_path else None,
    )
    save_data(output_dir, results_dict, mean_results_dict)


if __name__ == "__main__":
    app()
