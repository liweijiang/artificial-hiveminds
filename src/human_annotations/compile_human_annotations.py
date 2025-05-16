import os
import pandas as pd
from src.utils.main_utils import *


result_key_map = {
    "Response 1 is much better than Response 2": 2,
    "Response 1 is slightly better than Response 2": 1,
    "Response 1 and Response 2 are similar / it's hard to tell which one is better": 0,
    "Response 2 is slightly better than Response 1": -1,
    "Response 2 is much better than Response 1": -2,
}


def compile_rel(total_n_files: int, n_questions: int):
    base_dir = "data/adhoc_save/050725_human_annotations/v2_rel_030925/raw/"
    ref_data_path = f"data/adhoc_save/050725_human_annotations/v2_rel_030925/google_forms/google_form_to_queries.json"

    # Load reference data from google forms
    with open(ref_data_path, "r") as f:
        ref_data = json.load(f)

    all_records = []
    for i in range(1, total_n_files + 1):
        annotation_path = os.path.join(
            base_dir, f"rel_data_15_form_{i}.jsonl (Responses).xlsx")
        df_annotation = pd.read_excel(annotation_path)

        # Remove duplicate responses from the same Prolific ID, keeping the first response
        df_annotation = df_annotation.drop_duplicates(
            subset=["What's your **unique Prolific ID**?"],
            keep='first'
        )

        file_all_scores = {}
        for col_name in [f"Q{i}:" for i in range(1, n_questions + 1)]:
            qid = col_name.split(":")[0].strip()
            scores = df_annotation[col_name].tolist()
            file_all_scores[qid] = scores

        form_ref = ref_data[str(i - 1)]
        for qid in file_all_scores:
            qid_item = {
                "user_query": form_ref[qid]["user_query"],
                "response_1": form_ref[qid]["response_1"],
                "response_2": form_ref[qid]["response_2"],
                "human_labels": [result_key_map[score] for score in file_all_scores[qid]],
                "human_labels_raw": file_all_scores[qid],
                "form_id": i,
                "question_id": qid,
            }
            all_records.append(qid_item)

    save_path = f"data/adhoc_save/050725_human_annotations/v2_rel_030925/human_labels/all_records.jsonl"
    write_standard_data(all_records, save_path)


def compile_abs(total_n_files: int, n_questions: int):
    base_dir = "data/adhoc_save/050725_human_annotations/v2_abs_030925/raw/"
    ref_data_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/google_forms/google_form_to_queries.json"

    # Load reference data from google forms
    with open(ref_data_path, "r") as f:
        ref_data = json.load(f)

    all_records = []
    for i in range(1, total_n_files + 1):
        annotation_path = os.path.join(
            base_dir, f"abs_data_15_form_{i}.jsonl (Responses).xlsx")
        df_annotation = pd.read_excel(annotation_path)

        # Remove duplicate responses from the same Prolific ID, keeping the first response
        df_annotation = df_annotation.drop_duplicates(
            subset=["What's your **unique Prolific ID**?"],
            keep='first'
        )

        file_all_scores = {}
        for col_name in [f"Q{i}:" for i in range(1, n_questions + 1)]:
            qid = col_name.split(":")[0].strip()
            scores = df_annotation[col_name].tolist()
            file_all_scores[qid] = scores

        form_ref = ref_data[str(i - 1)]
        for qid in file_all_scores:
            qid_item = {
                "user_query": form_ref[qid]["user_query"],
                "response": form_ref[qid]["response"],
                "human_labels": file_all_scores[qid],
                "human_labels_raw": file_all_scores[qid],
                "form_id": i,
                "question_id": qid,
            }
            all_records.append(qid_item)

    save_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/human_labels/all_records.jsonl"
    write_standard_data(all_records, save_path)


if __name__ == "__main__":
    compile_rel(50, 10)
    compile_abs(50, 15)
