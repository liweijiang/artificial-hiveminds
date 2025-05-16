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


def load_human_labels():
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


def get_all_human_scores_sim_cluster(sim_method="middle_percentile", sim_method_param=0.1):
    all_human_scores_by_user_query = load_human_labels()

    if sim_method == "sim_threshold":
        all_data = []
        for user_query, data in all_human_scores_by_user_query.items():
            responses_1 = data["responses_1"]
            responses_2 = data["responses_2"]
            human_labels = data["human_labels"]
            human_labels_raw = data["human_labels_raw"]
            average_scores = data["average_scores"]

            for r_1, r_2, r_s, r_hl, r_hl_raw in zip(responses_1, responses_2, average_scores, human_labels, human_labels_raw):
                zero_count = r_hl.count(0)
                all_data.append({
                    "user_query": user_query,
                    "response_1": r_1,
                    "response_2": r_2,
                    "average_score": r_s,
                    "human_labels": r_hl,
                    "human_labels_raw": r_hl_raw,
                    "zero_count": zero_count
                })

        # Sort data by zero count in descending order
        all_data.sort(key=lambda x: x["zero_count"], reverse=True)

        # Calculate number of items to keep based on sim_method_param percentage
        num_items_to_keep = int(len(all_data) * (sim_method_param / 100))

        # Take top sim_method_param% of data with most zeros
        filtered_human_scores = all_data[:num_items_to_keep]

    if sim_method == "disagreement_top_percentile":
        all_data = []
        for user_query, data in all_human_scores_by_user_query.items():
            responses_1 = data["responses_1"]
            responses_2 = data["responses_2"]
            human_labels = data["human_labels"]
            average_scores = data["average_scores"]
            human_labels_raw = data["human_labels_raw"]

            for score, r_1, r_2, r_hl, r_hl_raw in zip(average_scores, responses_1, responses_2, human_labels, human_labels_raw):
                num_prefer_1 = sum(1 for score in r_hl if score > 0)
                num_prefer_2 = sum(1 for score in r_hl if score < 0)
                num_tie = sum(1 for score in r_hl if score == 0)

                max_num_prefer = max(num_prefer_1 + num_tie / 2, num_prefer_2 + num_tie / 2)
                percent_agreement = max_num_prefer / (num_prefer_1 + num_prefer_2 + num_tie)

                all_data.append({
                    "user_query": user_query,
                    "response_1": r_1,
                    "response_2": r_2,
                    "average_score": score,
                    "human_labels": r_hl,
                    "num_prefer_1": num_prefer_1,
                    "num_prefer_2": num_prefer_2,
                    "num_tie": num_tie,
                    "human_labels_raw": r_hl_raw,
                    "percent_agreement": percent_agreement
                })

        # Sort data by percent agreement in ascending order
        all_data.sort(key=lambda x: x["percent_agreement"], reverse=False)

        num_items_to_keep = int(len(all_data) * (sim_method_param / 100))
        filtered_human_scores = all_data[:num_items_to_keep]

    filtered_human_scores_map = {}
    for item in filtered_human_scores:
        user_query = item["user_query"]
        response_1 = item["response_1"]
        response_2 = item["response_2"]
        average_score = item["average_score"]

        if user_query not in filtered_human_scores_map:
            filtered_human_scores_map[user_query] = {
                "responses_1": [],
                "responses_2": [],
                "human_labels": [],
                "human_labels_raw": [],
                "average_scores": [],
            }

        filtered_human_scores_map[user_query]["responses_1"].append(response_1)
        filtered_human_scores_map[user_query]["responses_2"].append(response_2)
        filtered_human_scores_map[user_query]["human_labels"].append(
            item["human_labels"])
        filtered_human_scores_map[user_query]["human_labels_raw"].append(
            item["human_labels_raw"])
        filtered_human_scores_map[user_query]["average_scores"].append(
            average_score)

    return filtered_human_scores_map


def subselect_model_scores(human_scores, model_scores):
    user_query_response_map = {}
    for item in model_scores:
        user_query = item["user_query"]
        response = item["response"]
        user_query_response_map[user_query + response] = item

    selected_model_scores = {}
    for user_query in human_scores.keys():
        responses_1 = human_scores[user_query]["responses_1"]
        responses_2 = human_scores[user_query]["responses_2"]
        selected_model_scores[user_query] = {
            "user_query": user_query,
            "responses_1": [],
            "responses_2": [],
            "responses_1_scores": [],
            "responses_2_scores": [],
            "average_scores": [],
        }
        for response_1, response_2 in zip(responses_1, responses_2):
            selected_model_scores[user_query]["responses_1"].append(response_1)
            selected_model_scores[user_query]["responses_2"].append(response_2)
            selected_model_scores[user_query]["responses_1_scores"].append(
                user_query_response_map[user_query + response_1]["score"])
            selected_model_scores[user_query]["responses_2_scores"].append(
                user_query_response_map[user_query + response_2]["score"])
            selected_model_scores[user_query]["average_scores"].append(
                user_query_response_map[user_query + response_1]["score"] - user_query_response_map[user_query + response_2]["score"])
    return selected_model_scores


def analysis_rel(human_scores, model_name):
    model_scores = load_model_data(model_name)
    model_scores = subselect_model_scores(human_scores, model_scores)
    
    judge_scores_map = {
        "human": [],
        f"{model_name}": []
    }

    for user_query in human_scores.keys():
        human_avg_scores = human_scores[user_query]["average_scores"]
        model_avg_scores = model_scores[user_query]["average_scores"]

        judge_scores_map["human"].extend(human_avg_scores)
        judge_scores_map[model_name].extend(model_avg_scores)
        assert len(human_avg_scores) == len(model_avg_scores)

    # print("\nPairwise Spearman correlations between judges' score differences:")
    # print("-"*100)
    judge1 = model_name
    judge2 = "human"

    scores1 = judge_scores_map[judge1]
    scores2 = judge_scores_map[judge2]
    correlation, p_value = scipy.stats.spearmanr(scores1, scores2)
    # print(f"rho={correlation:.3f}, p={p_value:.3f}, n={len(scores1)} ({judge1})")

    print(
        judge1, f"rho={correlation:.3f}, p={p_value:.3f}, n={len(scores1)} ({judge1})")
    # print(f"{judge1},{correlation},{p_value},{len(scores1)}")

    return correlation, len(scores1)


def main_rel_analysis_combined_by_judge_types(sim_method="remove_outliers_tukey_iqr"):
    full_human_scores = load_human_labels()
    if sim_method == "sim_threshold":
        sim_method_params = [60, 65, 70, 75, 80, 85, 90, 95]

    colors = sns.color_palette("Set2")
    color_idx1 = 4
    color_idx2 = 0

    model_type_to_display_name = {
        "lm_judge": "LM Judge Scores",
        "reward_judge": "Reward Model Scores",
        "ppl_model": "LM Perplexities"
    }

    all_data_by_model_types = {}
    for model_type in model_types.keys():
        all_data_by_model_types[model_type] = {
            "scores": [],
            "ns": [],
            "model_names": [],
            "sim_method_params": [],
            "modes": [],
        }
        for sim_method_param in sim_method_params:
            human_scores = get_all_human_scores_sim_cluster(
                sim_method=sim_method, sim_method_param=sim_method_param)
            for model_name in model_types[model_type]:
                full_score, full_n = analysis_rel(full_human_scores, model_name)
                subset_score, subset_n = analysis_rel(human_scores, model_name)
                # print(f"{model_name} | Full: {full_score} | Subset: {subset_score}")

                all_data_by_model_types[model_type]["scores"].append(full_score)
                all_data_by_model_types[model_type]["ns"].append(full_n)
                all_data_by_model_types[model_type]["model_names"].append(model_name)
                all_data_by_model_types[model_type]["sim_method_params"].append(sim_method_param)
                all_data_by_model_types[model_type]["modes"].append("full")

                all_data_by_model_types[model_type]["scores"].append(subset_score)
                all_data_by_model_types[model_type]["ns"].append(subset_n)
                all_data_by_model_types[model_type]["model_names"].append(model_name)
                all_data_by_model_types[model_type]["sim_method_params"].append(sim_method_param)
                all_data_by_model_types[model_type]["modes"].append("subset")

        # Assuming data has already been loaded
        data = all_data_by_model_types[model_type]
        df = pd.DataFrame(data)

        # Get the unique parameters in reversed order
        sim_params_order = df['sim_method_params'].unique()[::-1]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.7), gridspec_kw={'width_ratios': [1, 6], 'wspace': 0.05})

        # Remove spines/edges from both subplots
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Define colors - making sure we use the correct ones
        subset_color = colors[color_idx1]  # The tan/yellowish color for subset
        full_color = colors[color_idx2]    # The blue color for full

        # SUBPLOT 1: (full mode) - Explicitly for parameter '3.0'
        full_data = df[(df['modes'] == 'full') & (df['sim_method_params'] == sim_method_params[0])].copy()
        full_data["sim_method_params"] = full_data["sim_method_params"].map(lambda x: "All" if x == sim_method_params[0] else x)

        # Check if we have data for the subset
        if not full_data.empty:
            # Create the subset bar plot
            sns.barplot(
                x=['All'],  # Force x-axis label
                y=[full_data['scores'].mean()],  # Use the mean value
                color=full_color,  # Use your tan color directly
                width=0.6,
                ax=ax1
            )

            # Add error bars for subset data if needed
            mean_val = full_data['scores'].mean()

            # Add annotation for subset bar
            ax1.text(
                0, mean_val + 0.01,
                f'{mean_val:.3f}'.lstrip("0").replace("-0", "-"),
                ha='center', va='bottom', fontsize=8
            )

        # SUBPLOT 2: (subset mode)
        subset_data = df[df['modes'] == 'subset'].copy()

        # Check if we have data for the full mode
        if not subset_data.empty:
            # Create a custom order of parameters to match what's shown in the original plot
            param_positions = {param: i for i, param in enumerate(sim_params_order)}

            # Create the bars manually to ensure correct colors and positions
            for param in sim_params_order:
                param_data = subset_data[subset_data['sim_method_params'] == param]
                if not param_data.empty:
                    pos = param_positions[param]
                    mean_val = param_data['scores'].mean()

                    # Draw the bar
                    ax2.bar(
                        pos,
                        mean_val,
                        width=0.6,
                        color=subset_color,  # Use the blue color
                        edgecolor='black',
                        linewidth=0
                    )

                    # Add text annotation
                    ax2.text(
                        pos, mean_val + 0.01,
                        f'{mean_val:.3f}'.lstrip("0").replace("-0", "-"),
                        ha='center', va='bottom', fontsize=8
                    )

            # Add x-label for the subset mode subplot
            # ax2.set_xlabel("Fence Factor: lower => remove more outliers", fontsize=8, labelpad=9)

            # Set the x-ticks for the full mode subplot
            ax2.set_xticks(range(len(sim_params_order)))
            ax2.set_xticklabels(sim_params_order)

            # Add n labels below x-ticks for subplot 2
            for param in sim_params_order:
                param_data = subset_data[subset_data['sim_method_params'] == param]
                if not param_data.empty:
                    pos = param_positions[param]
                    ax2.text(
                        pos, -0.11,
                        f'{param_data["ns"].iloc[0]}',
                        ha='center', va='top', fontsize=8,
                        transform=ax2.get_xaxis_transform()
                    )

        min_score = 0.0
        max_score = 0.5 + 0.01
        # Formatting for both subplots
        for idx, ax in enumerate([ax1, ax2]):
            if idx == 0:
                ax.set_ylabel("Spearman's Correlation Coeff. (ρ)", fontsize=10)
            else:
                ax.set_ylabel("", fontsize=10)

            ax.set_ylim(min_score, max_score)  # Adjusted to show n labels below 0
            ax.grid(axis='y', linestyle='--', alpha=0.3, which='major', color='gray')

            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: ""))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Set major grid lines every 0.1
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8, colors='white')
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color('lightgray')
            # ax.set_yticks([])

            if idx == 0:
                if model_type == "ppl_model":
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                    ax.tick_params(axis='y', labelsize=8, colors='black')
                    ax.set_yticks(np.arange(min_score, max_score, 0.1))
                else:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                    ax.tick_params(axis='y', labelsize=8, colors='white')
                    ax.set_yticks(np.arange(min_score, max_score, 0.1))

        # Add n label below x-tick for subplot 1
        if not full_data.empty:
            if model_type == "ppl_model":
                ax1.text(
                    0, -0.11,
                    f'n={full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )
            else:
                ax1.text(
                    0, -0.11,
                    f'{full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )

        # Customize x-axis for left subplot
        ax1.set_xlabel("")
        if subset_data.empty:
            ax1.set_xticks([])
            ax1.text(0.5, 0.5, 'No data for subset mode',
                    ha='center', va='center', transform=ax1.transAxes)

        # Add a title for the entire figure
        plt.suptitle(f"{model_type_to_display_name[model_type]}", fontsize=10, y=0.94)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        save_path = f'data/adhoc_save/050725_human_annotations/v2_rel_030925/plots/combined/{sim_method}/by_judge_types/{model_type}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()


def main_rel_analysis_combined_by_judge_types_disagreement(sim_method="disagreement_middle_percentile"):
    full_human_scores = load_human_labels()
    if sim_method == "disagreement_top_percentile":
        # sim_method_params = [16, 14, 12, 10, 8, 6, 4, 2]
        sim_method_params = [60, 65, 70, 75, 80, 85, 90, 95]

    colors = sns.color_palette("Set2")
    color_idx1 = 5
    color_idx2 = 0

    model_type_to_display_name = {
        "lm_judge": "LM Judge Scores",
        "reward_judge": "Reward Model Scores",
        "ppl_model": "LM Perplexities"
    }

    all_data_by_model_types = {}
    for model_type in model_types.keys():
        all_data_by_model_types[model_type] = {
            "scores": [],
            "ns": [],
            "model_names": [],
            "sim_method_params": [],
            "modes": [],
        }
        for sim_method_param in sim_method_params:
            human_scores = get_all_human_scores_sim_cluster(
                sim_method=sim_method, sim_method_param=sim_method_param)
            for model_name in model_types[model_type]:
                full_score, full_n = analysis_rel(full_human_scores, model_name)
                subset_score, subset_n = analysis_rel(human_scores, model_name)
                # print(f"{model_name} | Full: {full_score} | Subset: {subset_score}")

                all_data_by_model_types[model_type]["scores"].append(full_score)
                all_data_by_model_types[model_type]["ns"].append(full_n)
                all_data_by_model_types[model_type]["model_names"].append(model_name)
                all_data_by_model_types[model_type]["sim_method_params"].append(sim_method_param)
                all_data_by_model_types[model_type]["modes"].append("full")

                all_data_by_model_types[model_type]["scores"].append(subset_score)
                all_data_by_model_types[model_type]["ns"].append(subset_n)
                all_data_by_model_types[model_type]["model_names"].append(model_name)
                all_data_by_model_types[model_type]["sim_method_params"].append(sim_method_param)
                all_data_by_model_types[model_type]["modes"].append("subset")

        # Assuming data has already been loaded
        data = all_data_by_model_types[model_type]
        df = pd.DataFrame(data)

        # Get the unique parameters in reversed order
        sim_params_order = df['sim_method_params'].unique()[::-1]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.7), gridspec_kw={'width_ratios': [1, 6], 'wspace': 0.05})

        # Remove spines/edges from both subplots
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Define colors - making sure we use the correct ones
        subset_color = colors[color_idx1]  # The tan/yellowish color for subset
        full_color = colors[color_idx2]    # The blue color for full

        # SUBPLOT 1: (full mode) - Explicitly for parameter '3.0'
        full_data = df[(df['modes'] == 'full') & (df['sim_method_params'] == sim_method_params[0])].copy()
        full_data["sim_method_params"] = full_data["sim_method_params"].map(lambda x: "All" if x == sim_method_params[0] else x)

        # Check if we have data for the subset
        if not full_data.empty:
            # Create the subset bar plot
            sns.barplot(
                x=['All'],  # Force x-axis label
                y=[full_data['scores'].mean()],  # Use the mean value
                color=full_color,  # Use your tan color directly
                width=0.6,
                ax=ax1
            )

            # Add error bars for subset data if needed
            mean_val = full_data['scores'].mean()

            # Add annotation for subset bar
            ax1.text(
                0, mean_val + 0.01,
                f'{mean_val:.3f}'.lstrip("0").replace("-0", "-"),
                ha='center', va='bottom', fontsize=8
            )

        # SUBPLOT 2: (subset mode)
        subset_data = df[df['modes'] == 'subset'].copy()

        # Check if we have data for the full mode
        if not subset_data.empty:
            # Create a custom order of parameters to match what's shown in the original plot
            param_positions = {param: i for i, param in enumerate(sim_params_order)}

            # Create the bars manually to ensure correct colors and positions
            for param in sim_params_order:
                param_data = subset_data[subset_data['sim_method_params'] == param]
                if not param_data.empty:
                    pos = param_positions[param]
                    mean_val = param_data['scores'].mean()

                    # Draw the bar
                    ax2.bar(
                        pos,
                        mean_val,
                        width=0.6,
                        color=subset_color,  # Use the blue color
                        edgecolor='black',
                        linewidth=0
                    )

                    # Add text annotation
                    ax2.text(
                        pos, mean_val + 0.01,
                        f'{mean_val:.3f}'.lstrip("0").replace("-0", "-"),
                        ha='center', va='bottom', fontsize=8
                    )

            # Add x-label for the subset mode subplot
            # ax2.set_xlabel("Fence Factor: lower => remove more outliers", fontsize=8, labelpad=9)

            # Set the x-ticks for the full mode subplot
            ax2.set_xticks(range(len(sim_params_order)))
            ax2.set_xticklabels(sim_params_order)

            # Add n labels below x-ticks for subplot 2
            for param in sim_params_order:
                param_data = subset_data[subset_data['sim_method_params'] == param]
                if not param_data.empty:
                    pos = param_positions[param]
                    ax2.text(
                        pos, -0.11,
                        f'{param_data["ns"].iloc[0]}',
                        ha='center', va='top', fontsize=8,
                        transform=ax2.get_xaxis_transform()
                    )

        min_score = -0.2
        max_score = 0.5 + 0.01
        # Formatting for both subplots
        for idx, ax in enumerate([ax1, ax2]):
            if idx == 0:
                ax.set_ylabel("Spearman's Correlation Coeff. (ρ)", fontsize=10)
            else:
                ax.set_ylabel("", fontsize=10)

            ax.set_ylim(min_score, max_score)  # Adjusted to show n labels below 0
            ax.grid(axis='y', linestyle='--', alpha=0.3, which='major', color='gray')

            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: ""))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))  # Set major grid lines every 0.1
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8, colors='white')
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color('lightgray')
            # ax.set_yticks([])

            if idx == 0:
                if model_type == "ppl_model":
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                    ax.tick_params(axis='y', labelsize=8, colors='black')
                    ax.set_yticks(np.arange(min_score, max_score, 0.1))
                else:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                    ax.tick_params(axis='y', labelsize=8, colors='white')
                    ax.set_yticks(np.arange(min_score, max_score, 0.1))

        # Add n label below x-tick for subplot 1
        if not full_data.empty:
            if model_type == "ppl_model":
                ax1.text(
                    0, -0.11,
                    f'n={full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )
            else:
                ax1.text(
                    0, -0.11,
                    f'{full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )

        # Customize x-axis for left subplot
        ax1.set_xlabel("")
        if subset_data.empty:
            ax1.set_xticks([])
            ax1.text(0.5, 0.5, 'No data for subset mode',
                    ha='center', va='center', transform=ax1.transAxes)

        # Add a title for the entire figure
        plt.suptitle(f"{model_type_to_display_name[model_type]}", fontsize=10, y=0.94)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        save_path = f'data/adhoc_save/050725_human_annotations/v2_rel_030925/plots/combined/{sim_method}/by_judge_types/{model_type}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # main_rel_analysis_combined_by_judge_types(sim_method="sim_threshold")
    main_rel_analysis_combined_by_judge_types_disagreement(sim_method="disagreement_top_percentile")
