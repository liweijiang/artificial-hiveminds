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


def main_plot_entropy_rel(n_bins=15):
    full_human_scores = load_human_labels("rel")
    
    all_data = []
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["human_labels"]
        responses_1 = user_data["responses_1"]
        responses_2 = user_data["responses_2"]

        for scores, response_1, response_2 in zip(list_of_scores, responses_1, responses_2):
            entropy = shannon_entropy(scores, [-2, -1, 0, 1, 2])

            all_data.append({
                "user_query": user_query,
                "response_1": response_1,
                "response_2": response_2,
                "entropy": entropy,
            })

    all_data = pd.DataFrame(all_data)

    # Sort data by entropy
    all_data = all_data.sort_values('entropy')

    # Create histogram/bar chart
    plt.figure(figsize=(2.4, 3.2))
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    colors = sns.color_palette("Paired")

    plt.hist(all_data['entropy'], bins=n_bins, weights=np.ones(len(all_data))/len(all_data), alpha=0.7, color=colors[9], edgecolor=colors[9])
    plt.xlabel('Human Score Entropy', fontsize=10)
    plt.ylabel('% of (Query, Resp. 1, Resp. 2)', fontsize=10)
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    x_labels = np.linspace(all_data['entropy'].min(), all_data['entropy'].max(), 4)
    plt.xticks(x_labels, [f'{x:.1f}' for x in x_labels], fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    
    save_path = f'data/adhoc_save/050725_human_annotations/v2_rel_030925/plots/entropy_rel.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,
                dpi=600,
                bbox_inches='tight')
    plt.show()


def print_nth_bucket_examples_rel(n_bucket=0, n_bins=15):
    """Print query-response pairs that fall within the nth bucket of entropy values.
    
    Args:
        n_bucket (int): Which bucket to print examples from (0-indexed)
        n_bins (int): Number of bins to divide entropy values into (should match histogram)
    """
    full_human_scores = load_human_labels("rel")
    
    all_data = []
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["human_labels"]
        responses_1 = user_data["responses_1"]
        responses_2 = user_data["responses_2"]

        for scores, response_1, response_2 in zip(list_of_scores, responses_1, responses_2):
            entropy = shannon_entropy(scores, [-2, -1, 0, 1, 2])

            all_data.append({
                "user_query": user_query,
                "response_1": response_1,
                "response_2": response_2,
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
        print("~")
        print(f"Response 1: {row['response_1']}")
        print("~")
        print(f"Response 2: {row['response_2']}")
        print("~")
        print(f"Entropy: {row['entropy']:.3f}")
        print("-" * 80)
    
    print(f"Total examples: {len(bucket_examples)}")



def plot_score_distribution_rel(selected_query, selected_response_1, selected_response_2, cutoff=15):
    full_human_scores = load_human_labels("rel")
    
    user_query_to_scores = {}
    for user_query, user_data in full_human_scores.items():
        list_of_scores = user_data["human_labels"]
        responses_1 = user_data["responses_1"]
        responses_2 = user_data["responses_2"]

        for scores, response_1, response_2 in zip(list_of_scores, responses_1, responses_2):
            response_1 = response_1[:cutoff]
            response_2 = response_2[:cutoff]
            user_query_to_scores[f"{user_query} | {response_1} | {response_2}"] = scores

    selected_response_1 = selected_response_1[:cutoff]
    selected_response_2 = selected_response_2[:cutoff]

    selected_scores = user_query_to_scores[f"{selected_query} | {selected_response_1} | {selected_response_2}"]

    # Count frequency of each score
    score_counts = Counter(selected_scores)
    total_scores = len(selected_scores)
    score_percentages = [score_counts.get(i, 0) / total_scores * 100 for i in [-2, -1, 0, 1, 2]]

    # Create bar chart
    plt.figure(figsize=(3, 3.2))
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')

    # # Add grid for better readability
    # plt.grid(True, alpha=0.2)

    palette = sns.color_palette("Paired")
    colors = [palette[5], palette[4], palette[6], palette[2], palette[3]]

    # Create bar plot of score percentages with different colors
    for i, (score, percentage) in enumerate(zip([-2, -1, 0, 1, 2], score_percentages)):
        plt.bar(score, percentage, alpha=1, color=colors[i], edgecolor=colors[i])
        
    # plt.xlabel('Score', fontsize=12)
    # plt.ylabel('%', fontsize=10)
    
    # Set x-axis ticks to [1,2,3,4,5]
    plt.xticks([-2, -1, 0, 1, 2], ["strong pref. 2", "weak pref. 2", "neutral", "weak pref. 1", "strong pref. 1"], fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    
    save_path = f'data/adhoc_save/050725_human_annotations/v2_rel_030925/plots/score_distribution/{selected_query[:20]}_{selected_response_1[:20]}_{selected_response_2[:20]}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,
                dpi=600,
                bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    n_bins = 15
    main_plot_entropy_rel(n_bins)
    # print_nth_bucket_examples_rel(n_bucket=14, n_bins=n_bins)


# Query: Help me draft a paragraph as an expert consultant explaining TOEFL vs IELTS for international students.
# Response 1: Academic and General Training. It features a broader range of English accents and includes a face-to-face speaking interview, which some students may find more personal. Choosing the right test depends on your target institutions or visa requirements, comfort with test formats, and regional preferences. Both exams are respected internationally and designed to assess your English proficiency for academic and professional success.
# Response 2: When considering options for English language proficiency tests, international students often face the decision between the Test of English as a Foreign Language (TOEFL) and the International English Language Testing System (IELTS). Both exams are widely recognized and accepted by universities and institutions around the world, but they have distinct differences that can impact a student's choice. The TOEFL, administered by the Educational Testing Service (ETS), primarily focuses on American English and is structured into four sections: Reading, Listening, Speaking, and Writing. It is particularly known for its emphasis on academic contexts and its use of multiple-choice questions. On the other hand, the IELTS, jointly managed by the British Council, IDP: IELTS Australia, and Cambridge Assessment English, offers both an Academic and a General Training version. The Academic version is typically required for university admission and is more balanced in its use of British and American English. IELTS is known for its more diverse range of question types and its speaking test, which is conducted face-to-face with an examiner. Ultimately, the choice between TOEFL and IELTS should be based on the specific requirements of the institutions to which the student is applying, as well as the student's personal strengths and preferences in language skills.
# Entropy: 1.021

# Query: Write a movie title, and write a *literal* opposite of that movie title.
# Response 1: Movie Title: The Shawshank Redemption Literal Opposite: The Shawshank Imprisonment
# Response 2: The movie title is "A League of Their Own." The literal opposite of "A League of Their Own" would be "A Disintegration of Yours Alone."
# Entropy: 2.284

# Query: Write a fear-of-missing-out title including "You've Never Seen Anything Like This!"
# Response 1: Experience the Ultimate Adventure: You've Never Seen Anything Like This!
# Response 2: You Won't Believe What Everyone's Talking About: You've Never Seen Anything Like This!
# Entropy: 2.263

    plot_score_distribution_rel(selected_query="Help me draft a paragraph as an expert consultant explaining TOEFL vs IELTS for international students.",
                                selected_response_1="Academic and General Training. It features a broader range of English accents and includes a face-to-face speaking interview, which some students may find more personal. Choosing the right test depends on your target institutions or visa requirements, comfort with test formats, and regional preferences. Both exams are respected internationally and designed to assess your English proficiency for academic and professional success.",
                                selected_response_2="When considering options for English language proficiency tests, international students often face the decision between the Test of English as a Foreign Language (TOEFL) and the International English Language Testing System (IELTS). Both exams are widely recognized and accepted by universities and institutions around the world, but they have distinct differences that can impact a student's choice. The TOEFL, administered by the Educational Testing Service (ETS), primarily focuses on American English and is structured into four sections: Reading, Listening, Speaking, and Writing. It is particularly known for its emphasis on academic contexts and its use of multiple-choice questions. On the other hand, the IELTS, jointly managed by the British Council, IDP: IELTS Australia, and Cambridge Assessment English, offers both an Academic and a General Training version. The Academic version is typically required for university admission and is more balanced in its use of British and American English. IELTS is known for its more diverse range of question types and its speaking test, which is conducted face-to-face with an examiner. Ultimately, the choice between TOEFL and IELTS should be based on the specific requirements of the institutions to which the student is applying, as well as the student's personal strengths and preferences in language skills.")
    
    plot_score_distribution_rel(selected_query="Write a movie title, and write a *literal* opposite of that movie title.",
                                selected_response_1="Movie Title: The Shawshank Redemption Literal Opposite: The Shawshank Imprisonment",
                                selected_response_2='The movie title is "A League of Their Own." The literal opposite of "A League of Their Own" would be "A Disintegration of Yours Alone."')

    plot_score_distribution_rel(selected_query="Write a fear-of-missing-out title including \"You've Never Seen Anything Like This!\"",
                                selected_response_1="Experience the Ultimate Adventure: You've Never Seen Anything Like This!",
                                selected_response_2="You Won't Believe What Everyone's Talking About: You've Never Seen Anything Like This!")