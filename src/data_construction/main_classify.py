from src.utils.main_utils import *
from src.utils.chat_models import *
from data.eval.open_ended_taxonomy import *


def launch_batch_job_classify_open_ended_prompts():
    def load_prompt():
        prompt_path = "data/util_prompts/collect_open_ended_prompts/classify_open_ended_types.txt"
        with open(prompt_path, "r") as file:
            prompt = file.read()
        return prompt

    def load_taxonomy():
        taxonomy_string = ""
        for t in taxonomy:
            for tt in taxonomy[t]:
                taxonomy_string += f"- {tt}\n"
        return taxonomy_string

    def create_batch_input(data_path):
        judge_prompt_template = load_prompt()
        judge_prompt_template = judge_prompt_template.replace("{OPEN_ENDED_TAXONOMY}", load_taxonomy())
        data = load_standard_data(data_path)

        batch_input_data = []
        for d in tqdm(data):
            d_lm_judge_annotation = d["lm_judge_annotation"]
            user_query = d_lm_judge_annotation["original_query"] if d_lm_judge_annotation["revised_query"] is None else d_lm_judge_annotation["revised_query"]
            prompt = judge_prompt_template.replace("{USER_QUERY}", user_query)

            d["prompt"] = prompt

            batch_input_data.append({
                "custom_id": d["conversation_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}],
                    "max_tokens": 2048
                }
            })

        save_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_input.jsonl"
        write_standard_data(batch_input_data, save_path)


    def create_batch_job(data_path):
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        batch_input_file = client.files.create(
            file=open(data_path, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "classify open ended prompts categories"
            }
        )

        print(batch_job)
    data_path = "data/adhoc_save/011925_collect_open_ended_prompts/wildchat_filtered_full_lm_judge_annotated_clean_diverse.jsonl"
    create_batch_input(data_path)
    data_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_input.jsonl"
    create_batch_job(data_path)


def launch_batch_job_classify_open_ended_responses():
    return


if __name__ == "__main__":
    launch_batch_job_classify_open_ended_prompts()

