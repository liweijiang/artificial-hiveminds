import scipy
from src.utils.main_utils import *
import numpy as np
from src.adhoc.constants import *
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter


result_key_map = {
    "Response 1 is much better than Response 2": 2,
    "Response 1 is slightly better than Response 2": 1,
    "Response 1 and Response 2 are similar / it's hard to tell which one is better": 0,
    "Response 2 is slightly better than Response 1": -1,
    "Response 2 is much better than Response 1": -2,
}

model_types = {
    "lm_judge": lm_judge_models,
    "reward_judge": reward_models,
    "ppl_model": ppl_models_all
}


def load_model_data(model_name):
    _model_name = model_name.replace("/", "_")
    if model_name in reward_models:
        data_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/models/reward_models/{_model_name}_results.jsonl"
    elif model_name in lm_judge_models:
        data_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/models/{_model_name}/evaluate_results.jsonl"
    else:
        data_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/models/lm_ppls/{_model_name}_perplexity_scores.jsonl"

    data = load_standard_data(data_path, is_print=False)
    for i, item in enumerate(data):
        if model_name in reward_models:
            item["score"] = item["reward"]
        elif model_name in lm_judge_models:
            item["score"] = item["evaluation_score"]
        else:
            item["score"] = -item["conversation_perplexity"]
    return data


def load_human_labels(data_type):
    if data_type == "abs":
        data_path = f"data/adhoc_save/050725_human_annotations/v2_abs_030925/human_labels/all_records.jsonl"
        data = load_standard_data(data_path, is_print=False)

        all_human_scores_by_user_query = {}
        for item in data:
            user_query = item["user_query"]
            response = item["response"]
            human_labels = item["human_labels"]

            if user_query not in all_human_scores_by_user_query:
                all_human_scores_by_user_query[user_query] = {
                    "user_query": user_query,
                    "responses": [],
                    "raw_scores": [],
                    "average_scores": [],
                }

            all_human_scores_by_user_query[user_query]["responses"].append(
                response)
            all_human_scores_by_user_query[user_query]["raw_scores"].append(
                human_labels)
            all_human_scores_by_user_query[user_query]["average_scores"].append(
                np.mean(human_labels))
        return all_human_scores_by_user_query
    
    else:
        data_path = f"data/adhoc_save/050725_human_annotations/v2_rel_030925/human_labels/all_records.jsonl"
        data = load_standard_data(data_path, is_print=False)

        all_human_scores_by_user_query = {}
        for item in data:
            user_query = item["user_query"]
            response_1 = item["response_1"]
            response_2 = item["response_2"]
            human_labels = item["human_labels"]
            human_labels_raw = item["human_labels_raw"]
            if user_query not in all_human_scores_by_user_query:
                all_human_scores_by_user_query[user_query] = {
                    "user_query": user_query,
                    "responses_1": [],
                    "responses_2": [],
                    "human_labels": [],
                    "human_labels_raw": [],
                    "average_scores": [],
                }

            all_human_scores_by_user_query[user_query]["responses_1"].append(
                response_1)
            all_human_scores_by_user_query[user_query]["responses_2"].append(
                response_2)
            all_human_scores_by_user_query[user_query]["human_labels"].append(
                human_labels)
            all_human_scores_by_user_query[user_query]["human_labels_raw"].append(
                human_labels_raw)
            all_human_scores_by_user_query[user_query]["average_scores"].append(
                np.mean(human_labels))
        return all_human_scores_by_user_query


import numpy as np
from collections import Counter
import math

def shannon_entropy(annotations, possible_labels):
    """
    Calculate Shannon entropy for a list of annotation labels.
    
    Parameters:
    -----------
    annotations : list
        List of annotation values/labels.
    possible_labels : list
        List of all possible label values.
        
    Returns:
    --------
    float
        Shannon entropy value. Higher values indicate more disagreement.
    """
    # Count occurrences of each label
    counts = Counter(annotations)
    
    # Get total number of annotations
    total = len(annotations)
    
    # Calculate entropy
    entropy = 0
    for label in possible_labels:
        # Get count of this label (0 if not present)
        count = counts.get(label, 0)
        
        # Skip if count is 0 (0 * log(0) is defined as 0 in information theory)
        if count == 0:
            continue
            
        # Calculate probability of this label
        probability = count / total
        
        # Add to entropy calculation
        entropy -= probability * math.log2(probability)
    
    return entropy


def main_plot_entropy_abs(n_bins=15):
    full_human_scores = load_human_labels("abs")
    
    all_data = []
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["raw_scores"]
        responses = user_data["responses"]

        for scores, response in zip(list_of_scores, responses):
            entropy = shannon_entropy(scores, [1, 2, 3, 4, 5])

            all_data.append({
                "user_query": user_query,
                "response": response,
                "entropy": entropy,
            })

    all_data = pd.DataFrame(all_data)

    # Sort data by entropy
    all_data = all_data.sort_values('entropy')

    # Create histogram/bar chart
    plt.figure(figsize=(2.4, 3))
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    colors = sns.color_palette("hls", 8)

    plt.hist(all_data['entropy'], bins=n_bins, weights=np.ones(len(all_data))/len(all_data), alpha=0.75, color=colors[5], edgecolor=colors[5])
    plt.xlabel('Human Score Entropy', fontsize=10)
    plt.ylabel('% of Query-Response Pairs', fontsize=10)
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    x_labels = np.linspace(all_data['entropy'].min(), all_data['entropy'].max(), 4)
    plt.xticks(x_labels, [f'{x:.1f}' for x in x_labels], fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    
    save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/entropy_abs.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,
                dpi=600,
                bbox_inches='tight')
    plt.show()


def print_nth_bucket_examples_abs(n_bucket=0, n_bins=15):
    """Print query-response pairs that fall within the nth bucket of entropy values.
    
    Args:
        n_bucket (int): Which bucket to print examples from (0-indexed)
        n_bins (int): Number of bins to divide entropy values into (should match histogram)
    """
    full_human_scores = load_human_labels("abs")
    
    all_data = []
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["raw_scores"]
        responses = user_data["responses"]

        for scores, response in zip(list_of_scores, responses):
            entropy = shannon_entropy(scores, [1, 2, 3, 4, 5])

            all_data.append({
                "user_query": user_query,
                "response": response,
                "entropy": entropy,
            })

    all_data = pd.DataFrame(all_data)
    
    # Calculate bin edges using same parameters as histogram
    min_entropy = all_data['entropy'].min()
    max_entropy = all_data['entropy'].max()
    bin_edges = np.linspace(min_entropy, max_entropy, n_bins + 1)
    
    # Get examples from nth bucket
    bucket_min = bin_edges[n_bucket]
    bucket_max = bin_edges[n_bucket + 1]
    
    bucket_examples = all_data[
        (all_data['entropy'] >= bucket_min) & 
        (all_data['entropy'] < bucket_max)
    ]
    
    print(f"\nExamples from bucket {n_bucket} (entropy range: {bucket_min:.3f} - {bucket_max:.3f}):")
    print("-" * 80)
    
    for _, row in bucket_examples.iterrows():
        print(f"Query: {row['user_query']}")
        print(f"Response: {row['response']}")
        print(f"Entropy: {row['entropy']:.3f}")
        print("-" * 80)
    
    print(f"Total examples: {len(bucket_examples)}")



def plot_score_distribution_abs(selected_query, selected_response):
    full_human_scores = load_human_labels("abs")
    
    user_query_to_scores = {}
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["raw_scores"]
        responses = user_data["responses"]

        for scores, response in zip(list_of_scores, responses):
            user_query_to_scores[f"{user_query} | {response}"] = scores

    selected_scores = user_query_to_scores[f"{selected_query} | {selected_response}"]

    # Count frequency of each score
    score_counts = Counter(selected_scores)
    total_scores = len(selected_scores)
    score_percentages = [score_counts.get(i, 0) / total_scores * 100 for i in range(1, 6)]

    # Create bar chart
    plt.figure(figsize=(3, 2.4))
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    # # Add grid for better readability
    # plt.grid(True, alpha=0.2)

    palette = sns.color_palette("Paired")
    colors = [palette[5], palette[4], palette[6], palette[2], palette[3]]

    # Create bar plot of score percentages with different colors
    for i, (score, percentage) in enumerate(zip(range(1, 6), score_percentages)):
        plt.bar(score, percentage, alpha=1, color=colors[i], edgecolor=colors[i])
        
    plt.xlabel('Score', fontsize=12)
    # plt.ylabel('%', fontsize=10)
    
    # Set x-axis ticks to [1,2,3,4,5]
    plt.xticks(range(1, 6), fontsize=12)
    plt.yticks(fontsize=12)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    
    save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/score_distribution/{selected_query[:20]}_{selected_response[:20]}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,
                dpi=600,
                bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n_bins = 15
    main_plot_entropy_abs(n_bins)

    # bucket 2
    query_response_pairs = [
        [
            "What is Bukhara? Provide a paragraph-long explanation in layman's language.",
            "Bukhara is an ancient city located in present-day Uzbekistan, known for its rich history and cultural significance. It has been a major center of trade, religion, and learning along the Silk Road for over a thousand years. The city is famous for its stunning architecture, including beautiful mosques, madrasas (religious schools), and mausoleums, many of which date back to the medieval period. Bukhara's old town is a UNESCO World Heritage Site, attracting tourists from around the world who come to admire its well-preserved historical buildings and experience its traditional way of life. It’s a place where you can step back in time and feel the echoes of a bustling past filled with merchants, scholars, and travelers."
        ],
        
        [
            "Write a one- or two-sentence visual description of a game controller.",
            "The controller boasts a sleek, ergonomic design with two symmetrical analog sticks on the left, a curved D-pad just below them, and a vibrant cluster of four colored buttons on the right. Its smooth, matte finish is accented by glossy highlights around the shoulder and trigger buttons, suggesting both modern aesthetics and durable functionality."
        ]

    ]

    # # bucket 10
    # query_response_pairs = [
    #     [
    #         "Write a sentence about Sunday's fog by the ocean.",
    #         "On Sunday, the ocean was veiled in a thick blanket of fog, creating a mysterious and tranquil atmosphere along the shore."
    #     ],
    #     [
    #         "Name one meaning of life.",
    #         "One commonly cited perspective on the meaning of life, rooted in biology and evolution, is the **propagation of genes**—ensuring survival, reproduction, and the continuation of one's species. This view emphasizes life's inherent drive to adapt, thrive, and pass on genetic material to future generations. However, many philosophical, spiritual, and personal interpretations (e.g., seeking happiness, knowledge, or purpose) coexist, reflecting the complexity of the question."
    #     ]
    # ]


    # # bucket 14
    # query_response_pairs = [
    #     [
    #         "Write a pun about peanut.",
    #         'Why did the peanut go to the beach? Because it wanted to be a little more "grounded".'
    #     ],
    #     [
    #         "Which is the hottest English 5-letter word?",
    #         "A popular riddle answer is “HELLO,” because it literally contains the word “HELL” right inside it—hence it’s jokingly called the “hottest” five-letter word in English."
    #     ]
    # ]

    # # bucket 8
    # query_response_pairs = [
    #     [
    #         "Write a paragraph about how the internet shaped society.",
    #         "The internet has profoundly reshaped society by revolutionizing communication, commerce, and access to information. It has connected people across vast distances, fostering global collaboration and cultural exchange while enabling instant communication through email, social media, and video calls. This interconnectedness has also democratized knowledge, allowing individuals to learn from diverse sources and empowering marginalized voices to reach wider audiences. Additionally, the rise of e-commerce and digital platforms has transformed traditional business models, offering convenience and new opportunities for entrepreneurship. However, the internet's influence is not without challenges, such as misinformation, privacy concerns, and the digital divide, which highlight the need for responsible use and regulation. Overall, the internet has become a cornerstone of modern life, fundamentally altering how societies function and interact."
    #     ],
    #     [
    #         "Write a sentence about Sunday's fog by the ocean.",
    #         "On Sunday morning, the ocean lay cloaked in a silken blanket of fog, its gentle waves barely visible through the ethereal mist."
    #     ],
    #     [
    #         "Generate a description for 'Sticking to Cuteness: The Panda Way.'",
    #         "Get ready to embrace your inner cuddly creature with 'Sticking to Cuteness: The Panda Way'! These adorable panda-themed stickers are perfect for adding a touch of fluffy fun to your daily life. Whether you're a nature lover, a panda enthusiast, or just a fan of all things cute, these stickers will bring a smile to your face every time you see them. Stick them on your laptop, water bottle, notebook, or anywhere else you need a dose of cuteness. With their playful designs and vibrant colors, these stickers are sure to make your day a little brighter and remind you to embrace the simple joys of being a giant ball of fluff."
    #     ]
    # ]

    # # bucket 3
    # query_response_pairs = [
    #     [
    #         "Trivia Question: Which North American blue bird is known for its migratory habits and for being a symbol of happiness and good luck?",
    #         "The Eastern Bluebird (Sialia sialis) is known for its migratory habits and for being a symbol of happiness and good luck. It's a small bird with bright blue plumage on the back and wings, a reddish chest, and a white belly. They are common in the eastern and central parts of North America."
    #     ]
    # ]

    # # bucket 9
    # query_response_pairs = [
    #     [
    #         "Write a sentence about Sunday's fog by the ocean.",
    #         "The thick, rolling fog blanketed the coastline on Sunday morning, obscuring the usually vibrant ocean view and creating an eerie, yet tranquil atmosphere."
    #     ]
    # ]

    print_nth_bucket_examples_abs(n_bucket=2, n_bins=n_bins)

    for query, response in query_response_pairs:
        plot_score_distribution_abs(query, response)



