from src.utils.main_utils import write_standard_data
from tqdm import tqdm
from src.utils.main_utils import load_standard_data
from multiprocessing import Process
from src.utils.chat_models import GPTModel
import json
import pandas as pd
import os

def get_evaluate_inputs_save_path(data_path, judge_name, rubrics_version):
    save_path_components = data_path.split("/")
    save_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/models/gpt-4o-2024-11-20_{judge_name}/evaluate_inputs.jsonl"
    return save_path


def load_judge_prompt(judge_name):
    if judge_name == "hhh":
        judge_prompt_path = f"data/util_prompts/bootstrapped_judge/create_hhh_baseline_evaluation_template.txt"
    elif judge_name == "overall":
        judge_prompt_path = f"data/util_prompts/bootstrapped_judge/create_overall_baseline_evaluation_template.txt"
    else:
        raise ValueError(f"Invalid judge name: {judge_name}")
    with open(judge_prompt_path, "r") as file:
        judge_prompt = file.read()
    return judge_prompt


def format_judge_prompt(judge_prompt, user_query, response, prompt_to_rubrics):
    judge_prompt = judge_prompt.replace("{USER_REQUEST}", user_query)
    judge_prompt = judge_prompt.replace("{MODEL_RESPONSE}", response)
    if user_query in prompt_to_rubrics:
        rubrics = prompt_to_rubrics[user_query]["hard"]
        rubrics_str = "\n"
        for rubric in rubrics:
            rubrics_str += f"- {rubric['name']}: {rubric['description']}\n"
        judge_prompt = judge_prompt.replace("{EVALUATION_RUBRIC}", rubrics_str)
    return judge_prompt


def load_judge_rubrics(judge_name, rubrics_version="v1"):
    prompt_to_rubrics = {}
    if judge_name == "bootstrapped":
        rubrics_path = f"data/adhoc_save/011325_bootstrapped_judge/v1/hand_picked_evaluation_examples_bootstrapped_rubrics_{rubrics_version}.jsonl"
        rubrics = load_standard_data(rubrics_path)
        for rubric in rubrics:
            user_query = rubric["prompt"].split(
                "Now, let's begin the task.\n\nUser Request:")[-1].split("Output:")[0].strip().strip("\n")
            rubrics = rubric["rubrics"]
            prompt_to_rubrics[user_query] = rubrics
    return prompt_to_rubrics


def main_generate_judge_score_bootstrapped_adjusted(judge_name, data_path, rubrics_version="v1"):
    save_path = get_evaluate_inputs_save_path(data_path, judge_name, rubrics_version)
    if os.path.exists(save_path):
        return save_path

    judge_prompt = load_judge_prompt(judge_name)
    prompt_to_rubrics = load_judge_rubrics(judge_name, rubrics_version)
    data_to_evaluate = load_standard_data(data_path)

    formatted_data_to_evaluate = []
    for data in data_to_evaluate:
        user_query = data["user_query"]
        response = data["response"]

        formatted_judge_prompt = format_judge_prompt(
            judge_prompt, user_query, response, prompt_to_rubrics)

        formatted_data_to_evaluate.append(
            {"user_query": user_query, "response": response, "evaluation_input": formatted_judge_prompt})

    write_standard_data(formatted_data_to_evaluate, save_path, makedirs=True)
    return save_path


def parse_output(output):
    try:
        print("-"*100)
        print(output)
        output = output.strip("```")
        output = output.strip("json")
        output = output.strip("\n")
        output_json = json.loads(output)
        for d in output_json:
            item = output_json[d]
            if "score" not in item:
                return None
            if not isinstance(item["score"], (int, float)):
                return None
        return output_json
    except:
        print("Fail to parse output.")
        return None


def get_score(output_json):
    score = 0
    for d in output_json:
        item = output_json[d]
        score += float(item["score"])
    return score / len(output_json)


def main_evaluate(data_path, judge_name):
    # judge_model = get_chat_model("gpt-4o-2024-11-20", config={"num_tokens": 1024})
    model_config = {"model_name": "gpt-4o-2024-11-20",
                    "num_tokens": 1024, "temperature": 0, "top_p": 0.9}
    judge_model = GPTModel(model_config)

    save_path = data_path.replace("_inputs.jsonl", f"_results.jsonl")
    if os.path.exists(save_path):
        print(f"Loading existing data from {save_path}")
        data_to_evaluate = load_standard_data(save_path)
    else:
        print(f"Loading new data from {data_path}")
        data_to_evaluate = load_standard_data(data_path)

    for idx, data in tqdm(enumerate(data_to_evaluate), desc=f"Evaluating {judge_name}", total=len(data_to_evaluate)):
        if "evaluation_score" in data:
            print(f"Skipping data {idx} because it already has an evaluation score.")
            continue

        evaluation_input = data["evaluation_input"]
        output_json = None
        max_iter = 20
        _iter = 0
        while output_json is None and _iter < max_iter:
            output = judge_model.batch_generate([evaluation_input])[0]
            output_json = parse_output(output)
            print("-"*100)
            if _iter != 0:
                print(f"Retry {_iter} times, Output: {output_json}")
            _iter += 1

        if output_json is None:
            print(f"Fail to evaluate {judge_name} for {data['user_query']}.")
            continue

        score = get_score(output_json)

        data["evaluation_output"] = output_json
        data["evaluation_score"] = score

        write_standard_data(data_to_evaluate, save_path, makedirs=True)


def main(judge_name, data_path, rubrics_version="v1"):
    input_data_path = main_generate_judge_score_bootstrapped_adjusted(
        judge_name, data_path, rubrics_version)
    main_evaluate(input_data_path, judge_name)


if __name__ == '__main__':
    rubrics_version = "v1"
    data_path = "data/adhoc_save/050725_human_annotations/v2_abs_030925/human_labels/all_records.jsonl"

    processes = []
    judge_names = ["hhh", "overall"]
    for judge_name in judge_names:
        p = Process(target=main, args=(judge_name, data_path, rubrics_version,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
