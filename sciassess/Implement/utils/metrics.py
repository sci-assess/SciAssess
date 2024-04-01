import os
from pathlib import Path
import json
import re
from typing import Optional, Union, List, Any

from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

from sciassess.Implement.utils.postprocess import normalize, fuzzy_normalize_name, fuzzy_normalize_value


def fuzzy_match(s1: str, s2: str, **kwargs) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def fuzzy_compare_name(a: str, b: str, metric="EditDistance", **kwargs) -> Union[bool, float]:
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    a = a.strip()
    b = b.strip()

    if a == "" or b == "" and not a+b == "":
        return False
    if is_float(a) and is_float(b):
        return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)

    if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
        (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
        return True
    else:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / (len(a) + len(b))
        elif metric == "Word2Vec":
            pass


def fuzzy_compare_value(a: str, b: str, metric="EditDistance", **kwargs) -> Union[bool, float]:
    """
    Compare two strings with fuzzy matching.
    """

    def standardize_unit(s: str) -> str:
        """
        Standardize a (affinity) string to common units.
        """
        mark = "" if re.search(r"[><=]", s) is None else re.search(r"[><=]", s).group()
        unit = s.rstrip()[-2:]
        number = float(re.search(r"[\+\-]*[0-9.]+", s).group())

        if unit in ["µM", "uM"]:
            unit = "nM"
            number *= 1000
        elif unit in ["mM", "mm"]:
            unit = "nM"
            number *= 1000000

        if mark == "=":
            mark = ""
        return f"{mark}{number:.1f} {unit}"

    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    unit_str = ["nM", "uM", "µM", "mM", "%", " %", "wt.%", "at.%", "at%", "wt%"]
    nan_str = ["n/a", "nan", "na", "n.a.", "nd", "not determined", "not tested", "inactive"]
    a = a.strip()
    b = b.strip()
    if is_float(a) and is_float(b):
        return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)
    elif fuzzy_normalize_value(a) == "bal" or fuzzy_normalize_value(b) == "bal":
        return True
    elif fuzzy_normalize_value(a) == fuzzy_normalize_value(b):
        return True
    elif ((a[-2:] in unit_str or a[-1] in unit_str or a.split()[-1] in unit_str) and
          (b[-2:] in unit_str or b[-1] in unit_str or b.split()[-1] in unit_str)):
        a = standardize_unit(a)
        b = standardize_unit(b)
        return a == b
    elif a.lower() in nan_str and b.lower() in nan_str:
        return True
    if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
        (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
        return True
    else:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / (len(a) + len(b))
        elif metric == "Word2Vec":
            pass


def compare_molecule_similarity(smi1, smi2, **kwargs) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    mol1 = Chem.MolFromSmiles(re.sub(r'<.*>', '', str(smi1).strip("`")))
    mol2 = Chem.MolFromSmiles(re.sub(r'<.*>', '', str(smi2).strip("`")))

    if mol1 is None or mol2 is None:
        sim = 0.0
    else:
        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return {"score": sim}


def compare_molecule_strict(smi1, smi2, **kwargs) -> bool:
    from rdkit import Chem

    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    else:
        return Chem.MolToSmiles(Chem.RemoveHs(mol1)) == Chem.MolToSmiles(Chem.RemoveHs(mol2))


def match_list_bipartite(ind0: list, ind1: list, threshold: float = 0.9,
                         similarity_method=fuzzy_compare_name) -> dict:
    """
    Match the 2 list of string or tuple. Maybe indices of two dataframes.
    """
    from munkres import Munkres

    renames = {}
    similarities = np.array(np.ones([len(ind0) + 15, len(ind1) + 15]), dtype=np.float64)

    if similarity_method == fuzzy_compare_name:
        name2query = lambda name: name if type(name) != tuple else name[0] if len(name) == 1 or name[-1] == "" else name[-1]
        querys0 = [fuzzy_normalize_name(name2query(name)) for name in ind0]
        querys1 = [fuzzy_normalize_name(name2query(name)) for name in ind1]
    else:
        name2query = lambda name: name
        querys0 = ind0
        querys1 = ind1
    for i, name_i in enumerate(ind0):
        query_i = querys0[i]
        for j, name_j in enumerate(ind1):
            query_j = querys1[j]
            if query_i == "" or query_j == "":
                similarities[i, j] = 0
            result = similarity_method(query_i, query_j)
            if type(result) == bool:
                similarities[i, j] = 1 if result else 0
            elif type(result) == float:
                similarities[i, j] = result

    for k in range(15):
        for i in range(len(ind0)):
            similarities[i][len(ind1) + k] = threshold
        for j in range(len(ind1)):
            similarities[len(ind0) + k][j] = threshold
    dists = 1 - similarities
    # print(pd.DataFrame(dists, index=querys0 + ["v"] * 15, columns=querys1 + ["v"] * 15))

    # Kuhn-Munkres algorithm for useful solving the rectangular Assignment Problem
    mu = Munkres()
    indexes = mu.compute(dists.tolist())

    # 根据最优匹配下标输出映射
    for i, j in indexes:
        if (i < len(ind0)) and (j < len(ind1)):
            renames[name2query(ind1[j])] = name2query(ind0[i])
    return renames


def tableMatching(df_ref, df_prompt, index='Compound', compare_fields=[], record=True, file_name=None, **kwargs):
    assert len(df_ref) > 0, "Prompt table is empty."

    if df_prompt is None or len(df_prompt) == 0:
        return {"recall_field": 0.0, "recall_index": 0.0, "recall_value": 0.0, "recall_value_strict": 0.0,
                "accuracy_value": 0.0, "accuracy_value_strict": 0.0}
    metrics = {}
    index_names = ["Compound", "Name", "SMILES", "Nickname", "Substrate", "AlloyName"]

    if index not in [None, ""]:
        df_ref[index] = df_ref[index].astype(str)
        df_ref = df_ref.set_index(index)
        df_prompt[index] = df_prompt[index].astype(str)
        df_prompt = df_prompt.set_index(index)

    renames = match_list_bipartite(compare_fields, df_prompt.columns)
    renames = {key: value for key, value in renames.items() if key not in index_names}
    if len(renames) > 0:
        print("Find similar fields between answer and correct:", renames)
        df_prompt.rename(columns=renames, inplace=True)

    if index != "":
        renames = match_list_bipartite(df_ref.index, df_prompt.index)
        renames = {key: value for key, value in renames.items() if key not in index_names}
        if len(renames) > 0:
            print("Find similar indices between answer and correct:", renames)
            df_prompt.rename(index=renames, inplace=True)

    compare_fields_ = [col for col in compare_fields if
                       col not in [index] + ([index[0]] if type(index) == tuple else [])]
    metrics["recall_field"] = max(
        len([item for item in compare_fields_ if item in df_prompt.columns]) / len(compare_fields_), 1.0)
    metrics["recall_index"] = max(len([item for item in df_ref.index if item in df_prompt.index]) / df_ref.shape[0], 1.0)

    if record:
        from evals.record import record_match
        for col in compare_fields_:
            record_match(
                correct=col in df_prompt.columns,
                expected=col,
                picked=str(list(df_prompt.columns)),
                file_name=file_name,
                jobtype="match_field"
            )
        for ind in df_ref.index:
            record_match(
                correct=ind in df_prompt.index,
                expected=ind,
                picked=str(list(df_prompt.index)),
                file_name=file_name,
                jobtype="match_index"
            )

    match_score, total_match_score, smiles_match_score = 0.0, 0.0, 0.0
    N, M = len(df_ref.index), len(compare_fields_)
    for idx in df_ref.index:
        _total_matching = 1.0
        for col in compare_fields_:
            gtval = df_ref.loc[idx, col]
            gt = str(gtval.iloc[0]) if type(gtval) == pd.Series else str(gtval)
            try:
                pval = df_prompt.loc[idx, col]
                p = str(pval.iloc[0]) if type(pval) == pd.Series else str(pval)
            except:
                p = 'not found'

            _is_matching = fuzzy_compare_name(gt, p) if col != "SMILES" else compare_molecule_strict(gt, p)
            if col == "SMILES":
                smiles_match_score += float(_is_matching)
            if record:
                from evals.record import record_match
                record_match(
                    correct=_is_matching > 0,
                    expected=gt,
                    picked=p,
                    file_name=file_name,
                    jobtype="match_value" if col != "SMILES" else "match_SMILES"
                )
            _total_matching *= float(_is_matching)
            match_score += float(_is_matching) / M

        total_match_score += _total_matching
        _total_matching = 1.0

    metrics = {
        **metrics,
        "recall_value": match_score / N,
        "recall_value_strict": total_match_score / N,
        "accuracy_value": match_score / N * metrics["recall_index"],
        "accuracy_value_strict": total_match_score / N * metrics["recall_index"],
    }
    return metrics


EMBEDDING_MODEL = None


def load_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print('loading embedding model...')
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return EMBEDDING_MODEL


def cosine_similarity(sentence_ideal: str, sentence_pred: str) -> float:
    model = load_embedding_model()
    sentences = [sentence_ideal, sentence_pred]
    vecs = model.encode(sentences, show_progress_bar=False)
    vec1, vec2 = vecs
    dot_product = np.dot(vec1, vec2)
    cosine_sim = dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cosine_sim



def cosine_similarity_tuples(tuple_ideal: List[Any], tuple_pred: List[Any]) -> float:
    return min([cosine_similarity(word_ideal, word_pred) for word_ideal, word_pred in zip(tuple_ideal, tuple_pred)])


def match_bio_entities(words_ideal: list[Union[str, tuple]], words_pred: list[Union[str, tuple]], **kwargs):
    """
    Match the 2 list of biological words or tuples or triplets.
    """
    if len(words_ideal) == 0:
        print("Ideal list is empty.\n" + str(kwargs))
        return {"value_recall": 0.0, "f1_score": 0.0}
    if type(words_ideal[0]) == str:
        renames = match_list_bipartite(words_ideal, words_pred, similarity_method=cosine_similarity)
    elif type(words_ideal[0]) == tuple:
        renames = match_list_bipartite(words_ideal, words_pred, similarity_method=cosine_similarity_tuples)
    else:
        renames = {}

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(renames)
    FP = len(words_pred) - len(renames)
    FN = len(words_ideal) - len(renames)

    # Calculate Recall, Precision, and F1 Score
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        "value_recall": recall,
        "f1_score": f1_score
    }

