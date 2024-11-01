import os
import json
import typer
import pandas as pd
from typing import List, Optional
from metrics import jaccard_index_dictionary, jaccard_index_scispacy, cosine_similarity_biobert, machine_translation_metrics, bert_score_metric


# Create the Typer app
app = typer.Typer()

# Define available metrics
AVAILABLE_METRICS = [
    'jaccard_index_dictionary',
    'jaccard_index_scispacy',
    'cosine_similarity_biobert',
    'machine_translation_metrics',
    'bert_score_metric'
]

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

    # Check if 'generated' and 'original' columns exist
    if not {'generated', 'original'}.issubset(data.columns):
        raise ValueError("The file must contain 'generated' and 'original' columns or keys.")

    # Extract the 'generated' and 'original' columns as lists
    generated = data['generated'].tolist()
    original = data['original'].tolist()

    return generated, original


def run_evaluation(input_file, metrics, dict_path = None):
    original_texts, generated_texts = load_data(input_file)

    for metric in metrics:
        typer.echo(f"Computing {metric}...")
        if metric =='jaccard_index_dictionary':
                
            jacc_dict = jaccard_index_dictionary(original_texts, generated_texts, dict_path if dict_path else None)
        elif metric == 'jaccard_index_scispacy':
            jacc_scispacy = jaccard_index_scispacy(original_texts, generated_texts)
        elif metric == 'cosine_similarity_biobert':
            cosine_sim = cosine_similarity_biobert(original_texts, generated_texts)
        elif metric == 'machine_translation_metrics':
            bleu, cider, rouge = machine_translation_metrics(original_texts, generated_texts)
        elif metric == 'bert_score_metric':
            p, r, f1 = bert_score_metric(original_texts, generated_texts)
        else:
            typer.echo(f"Metric {metric} not available. Skipping...")

def save_data():
    pass




@app.command()
def main(
    input_file: str = typer.Argument(..., help="Path to the input file with the original and generated texts"),
    output_dir: str = typer.Argument(..., help="Path where the file with the metrics will be stored"),
    dict_path: Optional[str] = typer.Argument(..., help="Path to dictionary file with custom keywords"),

    metrics: Optional[List[str]] = typer.Option(
        None, 
        help=f"List of metrics to compute. Available options: {', '.join(AVAILABLE_METRICS)}. Defaults to all metrics."
    )
) -> None:

    
    output = run_evaluation(input_file, metrics, dict_path if dict_path else None)
    save_data(output, output_dir)



    
if __name__ == "__main__":
   app()