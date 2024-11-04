import os
import json
import typer
import pandas as pd
from typing import List, Optional
from pathlib import Path
from metrics import jaccard_index_dictionary, jaccard_index_scispacy, cosine_similarity_biobert, machine_translation_metrics, bert_score_metric


# Create the Typer app
app = typer.Typer()

# Define available metrics
AVAILABLE_METRICS = [
    'jaccard_index_dictionary',
   # 'jaccard_index_scispacy',
    'cosine_similarity_biobert',
    'machine_translation_metrics',
    'bert_score_metric'
]
      
def save_data(dir_path, results, mean_results):
   # Create the directory if it does not exist
    output_dir = Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "results.json"
    mean_output_file = output_dir / "results_averaged.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    with open(mean_output_file, 'w') as f:
            json.dump(mean_results, f, indent=4)


def load_data(input_file):
    # Check file extension
    _, file_extension = os.path.splitext(input_file)
    file_extension = file_extension.lower()

    if file_extension == '.csv':
        # Load CSV file
        data = pd.read_csv(input_file)
    elif file_extension == '.json':
        # Load JSON file
        with open(input_file, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file type. Only .csv and .json are allowed.")
    
    data.columns = data.columns.str.lower()
    # Check if 'generated' and 'original' columns exist
    if not {'generated', 'original'}.issubset(data.columns):
        raise ValueError("The file must contain 'generated' and 'original' columns or keys.")

    # Extract the 'generated' and 'original' columns as lists
    generated = data['generated'].tolist()
    original = data['original'].tolist()

    return generated, original


def run_evaluation(original_texts, generated_texts,metrics=None, dict_path = None):
    results = {}
    metrics = AVAILABLE_METRICS if metrics is None else metrics
    for metric in metrics:
        typer.echo(f"Computing {metric}...")

        if metric == 'jaccard_index_dictionary':  
            jacc_dict_score = jaccard_index_dictionary(original_texts, generated_texts, dict_path if dict_path else None)
            results['jaccard_index_dictionary'] = jacc_dict_score

        elif metric == 'jaccard_index_scispacy':
            jacc_scispacy_score = jaccard_index_scispacy(original_texts, generated_texts)
            results['jacc_scispacy_score'] = jacc_scispacy_score

        elif metric == 'cosine_similarity_biobert':
            cosine_sim = cosine_similarity_biobert(original_texts, generated_texts)
            results['cosine_similarity_biobert'] = cosine_sim

        elif metric == 'machine_translation_metrics':
            bleu, cider, rouge = machine_translation_metrics(original_texts, generated_texts)
            results['BLEU'] = bleu
            results['Cider'] = cider
            results['ROUGE'] = rouge

        elif metric == 'bert_score_metric':
            p, r, f1 = bert_score_metric(original_texts, generated_texts)
            results['Bertscore_p'] = p
            results['Bertscore_r'] = r
            results['Bertscore_f1'] = f1
            
        else:
            typer.echo(f"Metric {metric} not available. Skipping...")
        
    averaged_results = compute_mean_results(results)
    return results, averaged_results

def compute_mean_results(results):
    results_averaged = {}
    for k, v in results.items():
        # Calculate the mean of the list of values
        results_averaged[f"{k}_mean"] = (sum(v) / len(v)) if isinstance(v, list) and v else v  # Ensure v is a list and not empty before calculating mean
                            
    return results_averaged
  
@app.command()
def main(
    input_file: str = typer.Argument(..., help="Path to the input file with the original and generated texts"),
    output_dir: str = typer.Argument(..., help="Path where the file with the metrics will be stored"),
    dict_path: Optional[str] = typer.Option(None, "--dict-path", "-d", help="Path to dictionary file with custom keywords"),
    metrics: Optional[List[str]] = typer.Option(
        None, 
        "--metrics", "-m",
        help=f"List of metrics to compute. Available options: {', '.join(AVAILABLE_METRICS)}. Defaults to all metrics."
    )
) -> None:
    original_texts, generated_texts = load_data(input_file)
    results_dict, mean_results_dict = run_evaluation(original_texts, generated_texts, metrics if metrics else None, dict_path if dict_path else None)
    save_data(output_dir,results_dict, mean_results_dict)



    
if __name__ == "__main__":
   app()