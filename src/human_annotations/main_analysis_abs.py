import scipy
from src.utils.main_utils import *
import numpy as np
from src.adhoc.constants import *
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter


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


def get_all_human_scores_sim_cluster(sim_method="middle_percentile", sim_method_param=0.1):
    all_human_scores_by_user_query = load_human_labels()

    if sim_method == "sim_threshold":
        # Keep pairs of responses with similar scores for each query
        filtered_human_scores = []
        seen_pairs = set()  # Track seen response pairs
        seen_query_response = set()  # Track seen query+response pairs

        for user_query, data in all_human_scores_by_user_query.items():
            responses = data["responses"]
            raw_scores = data["raw_scores"]
            scores = data["average_scores"]

            # Find pairs of responses with scores within 0.1 of each other
            for i in range(len(scores)):
                for j in range(i+1, len(scores)):
                    if abs(scores[i] - scores[j]) < sim_method_param:
                        # Create unique identifier for this response pair
                        pair_id = tuple(sorted([responses[i], responses[j]]))

                        # Only add if we haven't seen this pair before
                        if pair_id not in seen_pairs:
                            seen_pairs.add(pair_id)

                            # Add both responses if not seen before
                            query_response_i = (user_query, responses[i])
                            query_response_j = (user_query, responses[j])

                            if query_response_i not in seen_query_response:
                                seen_query_response.add(query_response_i)
                                filtered_human_scores.append({
                                    "user_query": user_query,
                                    "response": responses[i],
                                    "raw_scores": raw_scores[i],
                                    "average_score": scores[i],
                                })

                            if query_response_j not in seen_query_response:
                                seen_query_response.add(query_response_j)
                                filtered_human_scores.append({
                                    "user_query": user_query,
                                    "response": responses[j],
                                    "raw_scores": raw_scores[j],
                                    "average_score": scores[j],
                                })

    if sim_method == "middle_percentile":
        # Keep responses within middle 80% of scores
        all_scores = []
        for user_query, data in all_human_scores_by_user_query.items():
            all_scores.extend(data["average_scores"])

        lower_bound = np.percentile(all_scores, (100 - sim_method_param) / 2)
        upper_bound = np.percentile(all_scores, (100 + sim_method_param) / 2)

        filtered_human_scores = []
        seen_query_response = set()

        for user_query, data in all_human_scores_by_user_query.items():
            responses = data["responses"]
            raw_scores = data["raw_scores"]
            scores = data["average_scores"]

            for i in range(len(scores)):
                if lower_bound <= scores[i] <= upper_bound:
                    query_response = (user_query, responses[i])

                    if query_response not in seen_query_response:
                        seen_query_response.add(query_response)
                        filtered_human_scores.append({
                            "user_query": user_query,
                            "response": responses[i],
                            "raw_scores": raw_scores[i],
                            "average_score": scores[i],
                        })

    if sim_method == "remove_outliers_tukey_iqr":
        # Calculate Q1, Q3 and IQR for average scores
        all_scores = []
        for user_query, data in all_human_scores_by_user_query.items():
            all_scores.extend(data["average_scores"])
            
        q1 = np.percentile(all_scores, 25)
        q3 = np.percentile(all_scores, 75)
        iqr = q3 - q1
        
        # Define bounds for outliers
        lower_bound = q1 - sim_method_param * iqr 
        upper_bound = q3 + sim_method_param * iqr

        filtered_human_scores = []
        seen_query_response = set()

        for user_query, data in all_human_scores_by_user_query.items():
            responses = data["responses"]
            raw_scores = data["raw_scores"]
            scores = data["average_scores"]

            for i in range(len(scores)):
                if lower_bound <= scores[i] <= upper_bound:
                    query_response = (user_query, responses[i])

                    if query_response not in seen_query_response:
                        seen_query_response.add(query_response)
                        filtered_human_scores.append({
                            "user_query": user_query,
                            "response": responses[i],
                            "raw_scores": raw_scores[i],
                            "average_score": scores[i],
                        })
    if sim_method == "remove_outliers_zscore":
        # Calculate z-scores for all scores
        all_scores = []
        for user_query, data in all_human_scores_by_user_query.items():
            all_scores.extend(data["average_scores"])
            
        mean = np.mean(all_scores)
        std = np.std(all_scores)
        
        filtered_human_scores = []
        seen_query_response = set()

        for user_query, data in all_human_scores_by_user_query.items():
            responses = data["responses"]
            raw_scores = data["raw_scores"]
            scores = data["average_scores"]

            for i in range(len(scores)):
                z_score = abs((scores[i] - mean) / std)
                if z_score <= sim_method_param:
                    query_response = (user_query, responses[i])

                    if query_response not in seen_query_response:
                        seen_query_response.add(query_response)
                        filtered_human_scores.append({
                            "user_query": user_query,
                            "response": responses[i],
                            "raw_scores": raw_scores[i],
                            "average_score": scores[i],
                        })

    if sim_method == "disagreement_middle_percentile":
        # Calculate entropy for each set of raw scores
        filtered_human_scores = []
        seen_query_response = set()
        all_entropies = []

        def compute_entropy(labels: List[int], all_classes: List[int]) -> float:
            """
            Compute the entropy of the class distribution.

            Args:
                labels (List[int]): The list of integer class labels.
                all_classes (List[int]): The list of all possible classes.

            Returns:
                float: The entropy of the distribution.
            """
            if not labels:
                return 0.0

            label_counts = Counter(labels)
            total = len(labels)

            entropy = 0.0
            for cls in all_classes:
                p = label_counts[cls] / total if cls in label_counts else 0
                if p > 0:
                    entropy -= p * math.log2(p)
            return entropy
        
        # First pass - calculate all entropies
        for user_query, data in all_human_scores_by_user_query.items():
            responses = data["responses"]
            raw_scores = data["raw_scores"]
            scores = data["average_scores"]
            
            for i, r_scores in enumerate(raw_scores):
                entropy = compute_entropy(r_scores, [1, 2, 3, 4, 5])
                all_entropies.append((entropy, user_query, responses[i], r_scores, scores[i]))

                # r_scores_grouped = []
                # for s in r_scores:
                #     if s in [1, 2]:
                #         r_scores_grouped.append(1)
                #     elif s in [4, 5]:
                #         r_scores_grouped.append(5)
                #     else:
                #         r_scores_grouped.append(3)
                # entropy = compute_entropy(r_scores_grouped, [1, 3, 5])
                # all_entropies.append((entropy, user_query, responses[i], r_scores, scores[i]))

        # Find entropy threshold for top sim_method_param percentile
        entropy_threshold = np.percentile([x[0] for x in all_entropies], (100-sim_method_param))
        
        # Second pass - filter based on entropy threshold
        for entropy, user_query, response, raw_score, score in all_entropies:
            if entropy >= entropy_threshold:
                query_response = (user_query, response)
                if query_response not in seen_query_response:
                    seen_query_response.add(query_response)
                    filtered_human_scores.append({
                        "user_query": user_query,
                        "response": response,
                        "raw_scores": raw_score,
                        "average_score": score,
                    })
            
    filtered_human_scores_map = {}
    for item in filtered_human_scores:
        user_query = item["user_query"]
        response = item["response"]
        raw_scores = item["raw_scores"]
        average_score = item["average_score"]

        if user_query not in filtered_human_scores_map:
            filtered_human_scores_map[user_query] = {
                "responses": [],
                "raw_scores": [],
                "average_scores": [],
            }

        filtered_human_scores_map[user_query]["responses"].append(response)
        filtered_human_scores_map[user_query]["raw_scores"].append(raw_scores)
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
        responses = human_scores[user_query]["responses"]
        selected_model_scores[user_query] = {
            "user_query": user_query,
            "responses": [],
            "average_scores": [],
        }
        for response in responses:
            selected_model_scores[user_query]["responses"].append(response)
            selected_model_scores[user_query]["average_scores"].append(
                user_query_response_map[user_query + response]["score"])
    return selected_model_scores


def analysis_abs(human_scores, model_name):
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

    print(judge1,f"rho={correlation:.3f}, p={p_value:.3f}, n={len(scores1)} ({judge1})")
    # print(f"{judge1},{correlation},{p_value},{len(scores1)}")

    return correlation, len(scores1)


def main_abs_analysis(sim_method="middle_percentile", sim_method_param=0.1, is_plot=True):
    full_human_scores = load_human_labels()
    human_scores = get_all_human_scores_sim_cluster(
        sim_method=sim_method, sim_method_param=sim_method_param)

    colors = sns.color_palette("Set2")
    color_idx1 = 2
    color_idx2 = 6
    for model_type in model_types.keys():
        all_models_scores = []
        for model_name in model_types[model_type]:
            print("-" * 100)
            full_score, full_n = analysis_abs(full_human_scores, model_name)
            subset_score, subset_n = analysis_abs(human_scores, model_name)
            # print(f"{model_name} | Full: {full_score} | Subset: {subset_score}")

            if full_score != None and subset_score != None:
                print(full_score - subset_score,
                      f"({model_name}) | n={full_n} | n={subset_n}")

            all_models_scores.append({
                "model_name": model_name,
                "full_score": full_score,
                "subset_score": subset_score,
                "full_n": full_n,
                "subset_n": subset_n
            })

        if is_plot:
            model_names = [score["model_name"].split(
                "/")[-1] for score in all_models_scores]
            full_scores = [score["full_score"] for score in all_models_scores]
            subset_scores = [score["subset_score"]
                             for score in all_models_scores]

            bar_width = 0.42  # Fixed bar width
            x = np.arange(len(model_names))

            # Calculate figure width based on number of bars and width
            total_plot_width = len(model_names) * \
                (bar_width * 2 + 0)  # .5 for spacing
            # Add margins, minimum 8 inches
            fig_width = max(total_plot_width + 0, 1)

            # Create figure with fixed height for the plot area (excluding labels)
            # Increased total height to accommodate labels
            fig = plt.figure(figsize=(fig_width, 3.5))
            # Add subplot with specific position to maintain fixed plot height
            # [left, bottom, width, height]
            ax = fig.add_axes([.05, .35, .93, .55])

            ax.bar(x - bar_width/2, full_scores, bar_width,
                   label='Full', color=colors[color_idx1])
            ax.bar(x + bar_width/2, subset_scores, bar_width,
                   label='Similar Subset', color=colors[color_idx2])

            # Add number labels on top of each bar
            for i in range(len(model_names)):
                ax.text(x[i] - bar_width/2, full_scores[i], f'.{str(full_scores[i])[2:5]}',
                        ha='center', va='bottom', fontsize=10)
                ax.text(x[i] + bar_width/2, subset_scores[i], f'.{str(subset_scores[i])[2:5]}',
                        ha='center', va='bottom', fontsize=10)

            ax.set_ylabel('Spearman Correlation')
            ax.set_xticks(x)
            ax.set_title(
                f'Human Scores vs. {model_types_display_names[model_type]}')
            ax.set_xticklabels(model_names, rotation=30, ha='right')
            # ax.legend()

            # Add padding to the top of the plot
            ymax = max(max(full_scores), max(subset_scores))
            ax.set_ylim(top=ymax * 1.15)  # Add 15% padding above highest bar

            save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/{sim_method}_{sim_method_param}/correlation_comparison_{model_type}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path,
                        dpi=600,
                        bbox_inches='tight')
            plt.close()


def main_abs_analysis_combined_by_params(sim_method="remove_outliers_tukey_iqr"):
    full_human_scores = load_human_labels()
    sim_method_params = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    colors = sns.color_palette("Set2")
    color_idx1 = 2
    color_idx2 = 6

    all_human_scores_for_all_params = {}
    for sim_method_param in sim_method_params:
        human_scores = get_all_human_scores_sim_cluster(sim_method=sim_method, sim_method_param=sim_method_param)
        all_human_scores_for_all_params[sim_method_param] = human_scores

        model_type_scores_per_param = {}
        for model_type in model_types.keys():
            model_type_scores_per_param[model_type] = {
                "full_scores": [],
                "subset_scores": [],
                "full_ns": [],
                "subset_ns": [],
                "model_names": [],
            }
            # all_models_scores = []
            for model_name in model_types[model_type]:
                full_score, full_n = analysis_abs(full_human_scores, model_name)
                subset_score, subset_n = analysis_abs(human_scores, model_name)
                # print(f"{model_name} | Full: {full_score} | Subset: {subset_score}")

                if full_score != None and subset_score != None:
                    print(full_score - subset_score,
                        f"({model_name}) | n={full_n} | n={subset_n}")

                # all_models_scores.append({
                #     "model_name": model_name,
                #     "full_score": full_score,
                #     "subset_score": subset_score,
                #     "full_n": full_n,
                #     "subset_n": subset_n
                # })

                model_type_scores_per_param[model_type]["full_scores"].append(full_score)
                model_type_scores_per_param[model_type]["subset_scores"].append(subset_score)
                model_type_scores_per_param[model_type]["full_ns"].append(full_n)
                model_type_scores_per_param[model_type]["subset_ns"].append(subset_n)
                model_type_scores_per_param[model_type]["model_names"].append(model_name)

        all_human_scores_for_all_params[sim_method_param] = model_type_scores_per_param

        data = model_type_scores_per_param
        
        # Create a list to store rows for the DataFrame
        rows = []
        # Process each judge type
        for judge_type in data.keys():
            # Process full scores
            for score in data[judge_type]['full_scores']:
                rows.append({
                    'Judge Type': judge_type,
                    'Score Type': 'Full Scores', 
                    'Score': score
                })
            
            # Process subset scores
            for score in data[judge_type]['subset_scores']:
                rows.append({
                    'Judge Type': judge_type,
                    'Score Type': 'Subset Scores',
                    'Score': score
                })

        # Create a DataFrame
        df = pd.DataFrame(rows)

        # Rename the judge types to make them more readable
        df['Judge Type'] = df['Judge Type'].replace({
            'lm_judge': 'LM Judge Scores',
            'reward_judge': 'Reward Model Scores',
            'ppl_model': 'LM Perplexities'
        })

        # Set the figure size
        plt.figure(figsize=(4, 3))

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")

        # Calculate mean scores for each group
        mean_scores = df.groupby(['Judge Type', 'Score Type'])['Score'].mean().reset_index()

        # Create the bar plot using seaborn
        ax = sns.barplot(
            x='Judge Type',
            y='Score',
            hue='Score Type',
            data=mean_scores,
            palette=[colors[color_idx1], colors[color_idx2]],
            width=0.8,
        )

        # Add a title and labels
        # plt.title('Comparison of Full and Subset Scores Across Judge Types', fontsize=16)
        # plt.xlabel('Judge Type', fontsize=14)
        plt.ylabel("Spearman's Correlation Coefficient", fontsize=8)

        # Customize the legend
        # plt.legend(title='Score Type', fontsize=12, title_fontsize=14)

        # Add count and mean annotations above each bar
        for i, judge_type in enumerate(['LM Judge', 'PPL Model', 'Reward Judge']):
            for j, score_type in enumerate(['Full Scores', 'Subset Scores']):
                # Filter data for this combination
                data_subset = df[(df['Judge Type'] == judge_type) & 
                                (df['Score Type'] == score_type)]
                
                count = len(data_subset)
                mean_val = data_subset['Score'].mean()
                
                # Calculate x position for the bar
                x_pos = i + [-0.2, 0.2][j]
                
                # # Add count annotation
                # plt.text(x_pos, mean_val/2,
                #         f'n={count}', ha='center', fontsize=8)

                # Add count annotation
                plt.text(x_pos, plt.ylim()[1] - 0.43,
                        f'n={count}', ha='center', fontsize=8)
                
                # Add mean annotation
                plt.text(x_pos, mean_val + (plt.ylim()[1] - plt.ylim()[0])*0.02,
                        f'{mean_val:.3f}', ha='center',
                        color='black', fontsize=8)

        # Format y-axis to show values with 2 decimal places
        plt.ylim(0, df['Score'].max() * 1.01)  # Add some space at the top
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        plt.xlabel("")
        plt.legend([],[], frameon=False)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/combined/{sim_method}/by_params/{sim_method_param}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main_abs_analysis_combined_by_judge_types(sim_method="remove_outliers_tukey_iqr"):
    full_human_scores = load_human_labels()
    if sim_method == "remove_outliers_tukey_iqr":
        sim_method_params = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    elif sim_method == "remove_outliers_zscore":
        sim_method_params = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    elif sim_method == "middle_percentile":
        sim_method_params = [40, 50, 60, 70, 80, 90]

    colors = sns.color_palette("Set2")
    color_idx1 = 2
    color_idx2 = 6

    model_type_to_display_name = {
        "lm_judge": "LM Judge Scores",
        "reward_judge": "Reward Model Scores",
        "ppl_model": "LM Perplexities"
    }

    # all_human_scores_for_all_params = {}
        # model_type_scores_per_param = {}
    
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
            human_scores = get_all_human_scores_sim_cluster(sim_method=sim_method, sim_method_param=sim_method_param)

            for model_name in model_types[model_type]:
                full_score, full_n = analysis_abs(full_human_scores, model_name)
                subset_score, subset_n = analysis_abs(human_scores, model_name)
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
        if sim_method == "middle_percentile":
            full_data = df[(df['modes'] == 'full') & (df['sim_method_params'] == 40)].copy()
            full_data["sim_method_params"] = full_data["sim_method_params"].map(lambda x: "All" if x == 40 else x)
        else:
            full_data = df[(df['modes'] == 'full') & (df['sim_method_params'] == 3.0)].copy()
            full_data["sim_method_params"] = full_data["sim_method_params"].map(lambda x: "All" if x == 3.0 else x)

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
                f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
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
                        f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
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
        max_score = 0.4 + 0.01
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

        # Set title for each subplot
        # ax1.set_title("All Resp.", fontsize=8)
        # ax2.set_title("Subset by Removing Outliers w/ Tukey's Fences", fontsize=8)

        # Add a title for the entire figure
        plt.suptitle(f"{model_type_to_display_name[model_type]}", fontsize=10, y=0.94)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save and show the plot
        save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/combined/{sim_method}/by_judge_types/{model_type}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()


def main_abs_analysis_combined_by_judge_types_disagreement(sim_method="disagreement_middle_percentile"):
    full_human_scores = load_human_labels()
    if sim_method == "disagreement_middle_percentile":
        sim_method_params = [2, 4, 6, 8, 10, 12, 14, 16]

    colors = sns.color_palette("Set2")
    color_idx1 = 1
    color_idx2 = 6 

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
            human_scores = get_all_human_scores_sim_cluster(sim_method=sim_method, sim_method_param=sim_method_param)

            for model_name in model_types[model_type]:
                full_score, full_n = analysis_abs(full_human_scores, model_name)
                subset_score, subset_n = analysis_abs(human_scores, model_name)

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

        data = all_data_by_model_types[model_type]
        df = pd.DataFrame(data)

        sim_params_order = df['sim_method_params'].unique()[::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.7), 
                                       gridspec_kw={'width_ratios': [1, 8], 'wspace': 0.05},
                                       sharey=True)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        subset_color = colors[color_idx1]
        full_color = colors[color_idx2]

        # SUBPLOT 1 (left): Full mode at param=0.02
        full_data = df[(df['modes'] == 'full') & (df['sim_method_params'] == sim_method_params[0])].copy()
        full_data["sim_method_params"] = full_data["sim_method_params"].map(lambda x: "All" if x == sim_method_params[0] else x)

        if not full_data.empty:
            mean_val = full_data['scores'].mean()
            pos_all = -1  # Position the "All" bar to the left of other bars

            # Adjust width to compensate for the subplot size difference (width_ratio is 1:8)
            # The visual width needs to be 8 times wider in the smaller subplot
            ax1.bar(
                pos_all,
                mean_val,
                width=0.6 * 8,  # Multiply by 8 to compensate for the width_ratio
                color=full_color,
                edgecolor='black',
                linewidth=0,
                align='center'
            )
            ax1.set_xticks([pos_all])
            ax1.set_xticklabels(['All'])

            # Adjust label position based on value sign
            label_offset = 0.01
            if mean_val < 0:
                # Place label below the bar for negative values
                ax1.text(
                    pos_all, mean_val - label_offset,
                    f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
                    ha='center', va='top', fontsize=8
                )
            else:
                # Place label above the bar for positive values
                ax1.text(
                    pos_all, mean_val + label_offset,
                    f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
                    ha='center', va='bottom', fontsize=8
                )

            if model_type == "ppl_model":
                ax1.text(
                    pos_all, -0.11,
                    f'n={full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )
            else:
                ax1.text(
                    pos_all, -0.11,
                    f'{full_data["ns"].iloc[0]}',
                    ha='center', va='top', fontsize=8,
                    transform=ax1.get_xaxis_transform()
                )


        # SUBPLOT 2 (right): Subset mode
        subset_data = df[df['modes'] == 'subset'].copy()

        if not subset_data.empty:
            param_positions = {param: i for i, param in enumerate(sim_params_order)}
            
            for param in sim_params_order:
                param_data = subset_data[subset_data['sim_method_params'] == param]
                if not param_data.empty:
                    pos = param_positions[param]
                    mean_val = param_data['scores'].mean()

                    ax2.bar(
                        pos,
                        mean_val,
                        width=0.6,  # This value is already 0.6, same as subplot 1
                        color=subset_color,
                        edgecolor='black',
                        linewidth=0,
                        align='center'
                    )

                    # Adjust label position based on value sign
                    label_offset = 0.01
                    if mean_val < 0:
                        # Place label below the bar for negative values
                        ax2.text(
                            pos, mean_val - label_offset,
                            f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
                            ha='center', va='top', fontsize=8
                        )
                    else:
                        # Place label above the bar for positive values
                        ax2.text(
                            pos, mean_val + label_offset,
                            f'{mean_val:.3f}'.lstrip('0').replace('-0', '-'),
                            ha='center', va='bottom', fontsize=8
                        )

            ax2.set_xticks(range(len(sim_params_order)))
            ax2.set_xticklabels(sim_params_order)

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
        max_score = 0.4 + 0.01

        for idx, ax in enumerate([ax1, ax2]):
            if idx == 0:
                ax.set_ylabel("Spearman's Correlation Coeff. (ρ)", fontsize=10)
            else:
                ax.set_ylabel("", fontsize=10)
                        
            ax.set_ylim(min_score, max_score)  # Adjusted to show n labels below 0
            ax.grid(axis='y', linestyle='--', alpha=0.3, which='major', color='gray')
            ax.set_yticks(np.arange(min_score, max_score, 0.1))  # Keep this
            ax.tick_params(axis='x', labelsize=8)

            if idx == 0 and model_type == "ppl_model":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                ax.tick_params(axis='y', labelsize=8, colors='black')  # Make sure they're visible
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                ax.tick_params(axis='y', labelsize=8, colors='white')
                
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color('lightgray')


        ax1.set_xlabel("")
        if subset_data.empty:
            ax1.set_xticks([])
            ax1.text(0.5, 0.5, 'No data for subset mode',
                     ha='center', va='center', transform=ax1.transAxes)

        # Set xlim for the first subplot to ensure the wider bar fits and is centered properly
        ax1.set_xlim(-5, 3)  # Wider limits to accommodate the wider bar

        plt.suptitle(f"{model_type_to_display_name[model_type]}", fontsize=10, y=0.94)
        plt.tight_layout()

        save_path = f'data/adhoc_save/050725_human_annotations/v2_abs_030925/plots/combined/{sim_method}/by_judge_types/{model_type}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # main_abs_analysis(sim_method="middle_percentile", sim_method_param=0.25, is_plot=True)
    # main_abs_analysis(sim_method="sim_threshold", sim_method_param=0.15, is_plot=True)

    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=3.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=2.5, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=2.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=1.5, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=1.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_tukey_iqr", sim_method_param=0.5, is_plot=True)

    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=3.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=2.5, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=2.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=1.5, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=1.0, is_plot=True)
    # main_abs_analysis(sim_method="remove_outliers_zscore", sim_method_param=0.5, is_plot=True)


    main_abs_analysis_combined_by_judge_types(sim_method="remove_outliers_zscore")
    # main_abs_analysis_combined_by_judge_types_disagreement(sim_method="disagreement_middle_percentile")

    # main_abs_analysis_combined_by_judge_types(sim_method="middle_percentile")

    # main_abs_analysis_disagreement(sim_method="disagreement_middle_percentile", sim_method_param=0.1, is_plot=True)



