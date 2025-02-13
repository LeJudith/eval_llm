import json
import spacy

from tqdm import tqdm

# import numpy as np
from bert_score import BERTScorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from src.extractor import KeywordExtractor

def jaccard_index_dictionary(original_texts, generated_texts, dict_path):
    """
    Calculate the Jaccard index between original and generated texts using a medical keyword dictionary.

    Args:
        original_texts (list of str): List of original texts.
        generated_texts (list of str): List of generated texts.
        dict_path (str): Path to the JSON file containing the keyword dictionary.

    Returns:
        list of float: List of Jaccard indices for each pair of original and generated texts.
    """
    with open(dict_path, "r") as JSON:
        keywords_dict = json.load(JSON)

    jaccard_indices = []
    keyword_extractor = KeywordExtractor(keywords_dict)

    for true_text, gen_text in tqdm(zip(original_texts, generated_texts)):
        keywords_true = keyword_extractor(true_text)
        auxiliary_extractor = KeywordExtractor(keywords_true)
        keywords_pred = auxiliary_extractor(gen_text)
        
        if len(keywords_true) == 0 and len(keywords_pred) == 0:
            jaccard_indices.append(1.0)
        elif len(keywords_true) == 0 or len(keywords_pred) == 0:
            jaccard_indices.append(0.0)
        else:
            intersection = len(set(keywords_true).intersection(set(keywords_pred)))
            union = len(set(keywords_true).union(set(keywords_pred)))
            jaccard_indices.append(intersection / union)

    return jaccard_indices

def jaccard_index_scispacy(reports_true, reports_pred):
    """
    Calculate the Jaccard index between true and predicted reports using SciSpacy.

    Args:
        reports_true (list of str): List of true reports.
        reports_pred (list of str): List of predicted reports.

    Returns:
        list of float: List of Jaccard indices for each pair of true and predicted reports.
    """
    keyword_extractor = spacy.load("en_core_sci_scibert")
    jaccard_indices = []

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):
        keywords_true = keyword_extractor(text_true)
        keywords_true = [token.text for token in keywords_true if token.pos_ == "NOUN"]

        auxiliary_extractor = KeywordExtractor(keywords_true, 0.6)
        keywords_pred = auxiliary_extractor(text_pred)

        if len(keywords_true) == 0 and len(keywords_pred) == 0:
            jaccard_indices.append(1.0)
        elif len(keywords_true) == 0 or len(keywords_pred) == 0:
            jaccard_indices.append(0.0)
        else:
            intersection = len(set(keywords_true).intersection(set(keywords_pred)))
            union = len(set(keywords_true).union(set(keywords_pred)))
            jaccard_indices.append(intersection / union)

    return jaccard_indices

def cosine_similarity_biobert(reports_true, reports_pred):
    """
    Calculate the cosine similarity between true and predicted reports using BioBERT embeddings.

    Args:
        reports_true (list of str): List of true reports.
        reports_pred (list of str): List of predicted reports.

    Returns:
        list of float: List of cosine similarity scores for each pair of true and predicted reports.
    """
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    cosine_similarities = []

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):
        embedding_true = model.encode(text_true, convert_to_tensor=True)
        embedding_pred = model.encode(text_pred, convert_to_tensor=True)
        cosine_similarities.append(util.pytorch_cos_sim(embedding_true, embedding_pred).item())

    return cosine_similarities


def machine_translation_metrics(reports_true, reports_pred):
    """
    Calculate machine translation metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, ROUGE, METEOR).

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        Dictionary containing per-report scores and averaged scores for BLEU, CIDEr, ROUGE, and METEOR.
    """
    
    scores = {
        "BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": [],
        "ROUGE-L": [], "METEOR": []
    }
    
    # Initialize scorers
    scorer_b, scorer_r, scorer_m = Bleu(), Rouge(), Meteor()
    
    # Compute BLEU, ROUGE, and METEOR per sentence
    for text_true, text_pred in tqdm(zip(reports_true, reports_pred), total=len(reports_true)):
        reference = {"id1": [text_true]}
        candidate = {"id1": [text_pred]}
       
        # Compute BLEU (returns 4 scores)
        score_b, _ = scorer_b.compute_score(reference, candidate)
        
        # Compute METEOR
        score_m, _ = scorer_m.compute_score(reference, candidate)
        
        # Compute ROUGE
        score_r, _ = scorer_r.compute_score(reference, candidate)
        
        # Store per-report scores
        scores["BLEU-1"].append(score_b[0])
        scores["BLEU-2"].append(score_b[1])
        scores["BLEU-3"].append(score_b[2])
        scores["BLEU-4"].append(score_b[3])
        scores["ROUGE-L"].append(score_r)
        scores["METEOR"].append(score_m)
    
    # Compute CIDEr at the dataset level (not per sentence)
    cider_scorer = Cider()
    reference_corpus = {i: [ref] for i, ref in enumerate(reports_true)}
    candidate_corpus = {i: [hyp] for i, hyp in enumerate(reports_pred)}
    cider_score, _ = cider_scorer.compute_score(reference_corpus, candidate_corpus)
    scores["CIDEr"]=cider_score
    
    return scores


# def machine_translation_metrics(reports_true, reports_pred):
#     """
#     Calculate machine translation metrics (BLEU, CIDEr, ROUGE, METEOR) for generated text.

#     Args:
#         reports_true (list of str): List of reference texts.
#         reports_pred (list of str): List of hypothesis texts.

#     Returns:
#         Tuple containing lists of scores for BLEU, CIDEr, ROUGE, and METEOR.
#     """
#     scores_b, scores_c, scores_r, scores_m = [], [], [], []
#     scorer_b, scorer_c, scorer_r, scorer_m = (Bleu(4), Cider(), Rouge(), Meteor())
    
#     #cider is computed across the entire dataset not each pair individually
#     score_c, _ = scorer_c.compute_score(reports_true, reports_pred)
#     scores_c.append(score_c)
    
#     for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):
#         reference = {"id1": [text_true]}
#         candidate = {"id1": [text_pred]}
       
#         score_b, _ = scorer_b.compute_score(reference, candidate)
#         score_m, _ = scorer_m.compute_score(reference, candidate)
#         score_r, _ = scorer_r.compute_score(reference, candidate)
#         scores_b.append(score_b[-1])
#         scores_m.append(score_m)
#         scores_r.append(score_r)
        
#     return scores_b, scores_c, scores_m, scores_r

def bert_score_metric(reports_true, reports_pred):
    """
    Compute BERTScore (Precision, Recall, F1) for generated text using DeBERTa.

    Args:
        reports_true (list of str): List of reference texts.
        reports_pred (list of str): List of hypothesis texts.

    Returns:
        Tuple containing lists of precision, recall, and F1 scores.
    """
    p_list, r_list, f1_list = [], [], []

    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40, device="cuda")

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):
        p, r, f1 = scorer.score([text_true], [text_pred])
        p_list.append(p.cpu().numpy().item())
        r_list.append(r.cpu().numpy().item())
        f1_list.append(f1.cpu().numpy().item())
    
    return p_list, r_list, f1_list