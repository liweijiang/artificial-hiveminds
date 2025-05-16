from src.utils.main_utils import *
from src.utils.chat_models import *
from data.eval.open_ended_taxonomy import *


def compile_subset_for_response_type_analysis(num_sample = 100):
    data_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_clean.jsonl"
    data = load_standard_data(data_path)

    all_categories_stats = {}
    all_categories_data = {}
    for d in data:
        categories = d["categories"]
        for c in categories:
            c_name = c["category"]
            c_type = c["type"]
            # Check if this category is a substring of any existing category
            if c_type == "predefined":
                if c_name not in all_categories_stats:
                    all_categories_stats[c_name] = 0
                all_categories_stats[c_name] += 1

                if c_name not in all_categories_data:
                    all_categories_data[c_name] = []
                all_categories_data[c_name].append(d)

    
    all_categories_data_subset = {}
    for c in all_categories_data:
        if c == "Opinions-Based Questions":
            continue
        all_categories_data_subset[c] = random.sample(all_categories_data[c], min(num_sample, len(all_categories_data[c])))

    all_data = []
    for c in all_categories_data_subset:
        print(c, ":", len(all_categories_data_subset[c]))
        for d in all_categories_data_subset[c]:
            d["user_query"] = d["messages"][0]["content"]
            all_data.append(d)

    save_path = f"data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_clean_subset-{num_sample}.jsonl"
    write_standard_data(all_data, save_path)


def parse_output(output):
    try:
        output = output.strip("```")
        output = output.strip("json")
        output = output.strip("\n")
        output_json = json.loads(output)
        return output_json
    except:
        print("Fail to parse output.")
        return None


if __name__ == "__main__":
    # compile_raw_output()
    # analyze_prompt_classification_taxonomy()

    compile_subset_for_response_type_analysis()

