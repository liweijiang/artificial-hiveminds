from src.utils.main_utils import *
from src.utils.chat_models import *
from data.eval.open_ended_taxonomy import *


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


def format_prompt(user_query):
    judge_prompt_template = load_prompt()
    judge_prompt_template = judge_prompt_template.replace(
        "{OPEN_ENDED_TAXONOMY}", load_taxonomy())
    prompt = judge_prompt_template.replace("{USER_QUERY}", user_query)
    return prompt


def check_parsed_format(output_json):
    if "categories" in output_json:
        return True
    else:
        return False


def re_annotate_with_lm_judge(user_query):
    annotator_model_name = "gpt-4o-2024-11-20"
    model_config = {"model_name": annotator_model_name, "num_tokens": 1024}
    judge_model = GPTModel(model_config)
    judge_prompt_template = load_prompt()
    judge_prompt_template = judge_prompt_template.replace(
        "{OPEN_ENDED_TAXONOMY}", load_taxonomy())
    prompt = judge_prompt_template.replace("{USER_QUERY}", user_query)
    return judge_model.batch_generate([prompt])[0]


def compile_raw_output():
    data_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_raw.jsonl"
    data = load_standard_data(data_path)
    print(data[0])

    data_path = "data/adhoc_save/011925_collect_open_ended_prompts/wildchat_filtered_full_lm_judge_annotated_clean_diverse.jsonl"
    input_data = load_standard_data(data_path)

    all_results = []
    for input_d, d in tqdm(zip(input_data, data), total=len(data)):
        user_query = input_d["messages"][0]["content"]
        output = d["response"]["body"]["choices"][0]["message"]["content"]
        output_json = parse_output(output)

        retry_count = 0
        while output_json is None or "categories" not in output_json:
            print(f"retry {retry_count}")
            output = re_annotate_with_lm_judge(user_query)
            output_json = parse_output(output)
            retry_count += 1
            if retry_count > 10:
                break

        if output_json is None or "categories" not in output_json:
            print(f"Failed to parse output for {input_d['conversation_id']}")
            continue

        categories = output_json["categories"]
        input_d["categories"] = categories
        all_results.append(input_d)

    output_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_clean.jsonl"
    write_standard_data(all_results, output_path)


def parse_new_categories(all_new_categories):
    all_parsed_new_categories = []
    for c in all_new_categories:
        if "New Category:" in c:
            c = c.replace("New Category:", "").strip()
        if "New -" in c:
            c = c.replace("New -", "").strip()
        if "(New)" in c:
            c = c.replace("(New)", "").strip()
        if "New" in c:
            c = c.replace("New", "").strip()
        if ":" in c:
            c = c.replace(":", "").strip()

        if c not in all_parsed_new_categories:
            all_parsed_new_categories.append(c)
    return all_parsed_new_categories


def analyze_prompt_classification_taxonomy():
    data_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_clean.jsonl"
    data = load_standard_data(data_path)

    all_categories_stats = {}
    all_new_categories = []
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
            else:
                all_new_categories.append(c_name)
    all_names = list(all_categories_stats.keys())
    for i, name1 in enumerate(all_names):
        for name2 in all_names[i+1:]:
            if name1.lower() in name2.lower() or name2.lower() in name1.lower():
                shorter = name1 if len(name1) < len(name2) else name2
                longer = name2 if len(name1) < len(name2) else name1
                if longer in all_categories_stats and shorter in all_categories_stats:  # Check if already merged
                    all_categories_stats[shorter] += all_categories_stats[longer]
                    del all_categories_stats[longer]

    for c in all_categories_stats:
        print(f"{c},{100 * all_categories_stats[c]/len(data):.5f}")

    all_parsed_new_categories = parse_new_categories(all_new_categories)

    for c in all_parsed_new_categories:
        print(c)


def main_parse_new_categories():
    data_path = "data/adhoc_save/020925_classify_open_ended_prompts/classify_open_ended_types_batch_output_clean.jsonl"
    data = load_standard_data(data_path)

    all_new_categories = []
    for d in data:
        categories = d["categories"]
        for c in categories:
            c_name = c["category"]
            c_type = c["type"]
            # Check if this category is a substring of any existing category
            if c_type == "predefined":
                continue
            else:
                all_new_categories.append(c_name)

    all_parsed_new_categories = parse_new_categories(all_new_categories)

    # Create word cloud from parsed new categories
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Join all categories into a single string with spaces
    text = " ".join(all_parsed_new_categories)
    text = text.replace("questions", "").replace(
        "question", "").replace("Questions", "").replace("Question", "")

    # Generate word cloud
    wordcloud = WordCloud(width=1600,
                          height=800,
                          colormap="managua", 
                          background_color='white',
                          min_font_size=15).generate(text)

    # Display the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the word cloud
    save_path = "data/adhoc_save/020925_classify_open_ended_prompts/new_categories_wordcloud.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)
    plt.close()


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

    main_parse_new_categories()
