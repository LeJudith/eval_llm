import json
import spacy

from tqdm import tqdm
import numpy as np
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
        keywords_pred = auxiliary_extractor(
            gen_text
        )  # use keyword_extractor instead of auxiliary_extractor if you want all keywods that are found

        if len(keywords_true) == 0 and len(keywords_pred) == 0:
            jaccard_indices.append(
                1.0
            )  # If both are empty, consider it a perfect match
        elif len(keywords_true) == 0 or len(keywords_pred) == 0:
            jaccard_indices.append(0.0)  # If one is empty, Jaccard index is zero
        else:
            intersection = len(set(keywords_true).intersection(set(keywords_pred)))
            union = len(set(keywords_true).union(set(keywords_pred)))
            jaccard_indices.append(intersection / union)
            # print('Jaccard index: {intersection / union}', intersection / union)

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
            jaccard_indices.append(
                1.0
            )  # If both are empty, consider it a perfect match
        elif len(keywords_true) == 0 or len(keywords_pred) == 0:
            jaccard_indices.append(0.0)  # If one is empty, Jaccard index is zero
        else:
            intersection = len(set(keywords_true).intersection(set(keywords_pred)))
            union = len(set(keywords_true).union(set(keywords_pred)))
            jaccard_indices.append(intersection / union)

    #print(np.mean(np.array(jaccard_indices)))
    return jaccard_indices


def cosine_similarity_biobert(reports_true, reports_pred):
    """
    Calculate the BLEU score for the given references and hypotheses.

    Args:
        references (list of list of str): List of reference texts.
        hypotheses (list of str): List of hypothesis texts.

    Returns:
        float: The BLEU score.
    """
    model = SentenceTransformer(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    cosine_similarities = []

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):

        embedding_true = model.encode(text_true, convert_to_tensor=True)
        embedding_pred = model.encode(text_pred, convert_to_tensor=True)

        cosine_similarities.append(
            util.pytorch_cos_sim(embedding_true, embedding_pred).item()
        )
       # print(np.mean(np.array(cosine_similarities)))

    return cosine_similarities


def machine_translation_metrics(reports_true, reports_pred):
    """
    Calculate the CIDEr score for the given references and hypotheses.

    Args:
        references (list of list of str): List of reference texts.
        hypotheses (list of str): List of hypothesis texts.

    Returns:
        float: The CIDEr score.
    """
    scores_b, scores_c, scores_r = [], [], []  # , []
    scorer_b, scorer_c, scorer_r = Bleu(4), Cider(), Rouge()  # Meteor(),

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):

        reference = {"id1": [text_true]}
        candidate = {"id1": [text_pred]}

        score_b, _ = scorer_b.compute_score(reference, candidate)
        score_c, _ = scorer_c.compute_score(reference, candidate)
        # score_m, _ = scorer_m.compute_score(reference, candidate)
        score_r, _ = scorer_r.compute_score(reference, candidate)

        scores_b.append(score_b[-1])
        scores_c.append(score_c)
        # scores_m.append(score_m)
        scores_r.append(score_r)

        # print(
        #     f"Score: {{"
        #     f"{np.mean(np.array(scores_b)) * 100}, "
        #     f"{np.mean(np.array(scores_c)) * 100}, "
        #     #  f"{np.mean(np.array(scores_m)) * 100}, "
        #     f"{np.mean(np.array(scores_r)) * 100}"
        #     f"}}"
        # )

    return scores_b, scores_c, scores_r  # scores_m


def bert_score_metric(reports_true, reports_pred):
    p_list, r_list, f1_list = [], [], []

    scorer = BERTScorer(
        model_type="microsoft/deberta-xlarge-mnli", num_layers=40, device="cuda"
    )

    for text_true, text_pred in tqdm(zip(reports_true, reports_pred)):

        p, r, f1 = scorer.score([text_true], [text_pred])

        p_list.append(p.cpu().numpy().item())
        r_list.append(r.cpu().numpy().item())
        f1_list.append(f1.cpu().numpy().item())

        # print(
        #     np.mean(np.array(p_list)),
        #     np.mean(np.array(r_list)),
        #     np.mean(np.array(f1_list)),
        # )

    return p_list, r_list, f1_list
