import os
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import numpy.typing as npt
from src.adhoc.constants import model_names, reasoning_model_names, selected_model_names
from src.utils.main_utils import load_standard_data
from tqdm import tqdm
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


def load_model_data(model_name: str):
    model_name = model_name.replace("/", "_")
    embeddings_path = f"data/adhoc_save/050725_organize_model_generations/embeddings/{model_name}.jsonl"

    try:
        data = load_standard_data(embeddings_path, is_print=False)
        return data
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return []


def load_all_data_for_clustering(is_reload: bool = True):
    if is_reload:
        with open("data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/all_data_for_clustering.json", "r") as f:
            return json.load(f)

    all_data = {}
    all_data_responses = {}
    for model_name in tqdm(selected_model_names):
        data = load_model_data(model_name)

        for d in data:
            p = d["prompt"]
            if p not in all_data:
                all_data[p] = {"embeddings": [], "models": []}
                all_data_responses[p] = []

            all_data[p]["embeddings"].extend(d["embeddings"])
            all_data[p]["models"].extend([model_name] * len(d["responses"]))
            all_data_responses[p].extend(d["responses"])

    with open("data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/all_data_for_clustering.json", "w") as f:
        json.dump(all_data, f)

    with open("data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/all_data_responses_for_clustering.json", "w") as f:
        json.dump(all_data_responses, f)
    return all_data


def get_unique_model_counts_per_prompt():
    num_items_per_model = 50

    # run_clustering()

    print("Loading data...")
    start_time = time.time()
    all_data = load_all_data_for_clustering(is_reload=True)
    print(f"Data loaded. Time taken: {time.time() - start_time} seconds")

    all_unique_model_counts_per_prompt = {j+1: []
                                          for j in range(num_items_per_model)}
    for p, d in tqdm(all_data.items(), total=len(all_data)):
        d_models = d["models"]
        d_embeddings = np.array(d["embeddings"])

        # Calculate pairwise distances between all embeddings
        distances = cdist(d_embeddings, d_embeddings, metric='cosine')

        # Set diagonal to infinity to exclude self-pairs
        np.fill_diagonal(distances, np.inf)

        # For each embedding, find its closest num_items_per_model neighbors
        # closest_models = []
        for j in range(1, num_items_per_model+1):
            local_unique_model_counts = []
            for i in range(len(d_embeddings)):
                # Get distances from current embedding to all others
                dist_row = distances[i]

                # Get indices of num_items_per_model closest embeddings
                closest_indices = np.argpartition(dist_row, j)[:j]

                # Get the models for these closest embeddings
                models_for_closest = [d_models[idx] for idx in closest_indices]

                # Count unique source models
                unique_models = len(set(models_for_closest))
                # closest_models.append(unique_models)
                local_unique_model_counts.append(unique_models)
                # print(f"Prompt: {p}")
                # print(f"Embedding {i} closest {num_items_per_model} responses come from {unique_models} different models")
                # print(f"Models: {set(models_for_closest)}")
                # print()
            all_unique_model_counts_per_prompt[j].append(
                np.mean(local_unique_model_counts))

    print(all_unique_model_counts_per_prompt)

    # Average number of source models across all embeddings
    # avg_source_models = np.mean(closest_models)
    # print(f"Average number of source models in closest {num_items_per_model} responses: {avg_source_models:.2f}")
    # all_unique_model_counts.append(all_unique_model_counts)

    # print(f"Overall average number of source models: {np.mean(all_unique_model_counts):.2f}")

    # Save the unique model counts data
    save_path = "data/adhoc_save/050925_final_analysis_model_gen/unique_model_counts/unique_model_counts_per_prompt_grouped_by_all_embeddings.json"
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_unique_model_counts_per_prompt, f)


def plot_unique_model_counts_per_prompt():
    data_path = "data/adhoc_save/021625_model_similarity_plots/cross_models/unique_model_counts_per_prompt_grouped_by_all_embeddings.json"
    with open(data_path, "r") as f:
        all_unique_model_counts_per_prompt = json.load(f)

    selected_unique_model_counts_per_prompt = {}
    for j in all_unique_model_counts_per_prompt:
        print(
            f"Average number of source models in closest {j} responses: {np.mean(all_unique_model_counts_per_prompt[j]):.2f}")
        j = int(j)
        if (j % 5) == 0 and j > 0:
            selected_unique_model_counts_per_prompt[j] = all_unique_model_counts_per_prompt[str(
                j)]

    # Create box plots for unique model counts
    plt.figure(figsize=(8, 4.5))

    # Convert data to format for box plot
    plot_data = [selected_unique_model_counts_per_prompt[i]
                 for i in selected_unique_model_counts_per_prompt]

    # (75, 165, 157)
    box_color = (0, 0, 0)
    box_color = [x/255 for x in box_color]
    box_linewidth = 1.2

    median_color = (237, 110, 87)
    median_color = [x/255 for x in median_color]

    # Create box plot with custom colors
    plt.boxplot(plot_data, medianprops=dict(color=median_color, linewidth=2),
                boxprops=dict(color=box_color, alpha=1,
                              linewidth=box_linewidth),
                whiskerprops=dict(color=box_color, alpha=1,
                                  linewidth=box_linewidth),
                capprops=dict(color=box_color, alpha=1,
                              linewidth=box_linewidth),
                flierprops=dict(marker='o', markeredgecolor=box_color,
                                markersize=8, alpha=1, linewidth=box_linewidth),
                widths=0.6)  # markerfacecolor='white',

    # Add individual points with jitter
    for i, data in enumerate(plot_data, 1):
        # Create random x-coordinates centered on the box plot position
        x = np.random.normal(i, 0.05, size=len(data))
        plt.scatter(x, data, alpha=0.5, color=(
            75/255, 165/255, 157/255), s=15)  # (0.68, 0.85, 0.9)

    plt.title(
        '# Unique Source Models for N Closest Responses of Each Open-Ended Query', fontsize=11)
    plt.xlabel('N Closest Responses', fontsize=11)
    plt.ylabel('# Unique Source Models', fontsize=11)

    # Set x-axis labels
    plt.xticks(range(1, len(selected_unique_model_counts_per_prompt) + 1),
               [str(j) for j in selected_unique_model_counts_per_prompt], fontsize=11)

    plt.yticks(fontsize=11)
    plt.ylim(1.1, 11.2)

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig("data/adhoc_save/021625_model_similarity_plots/cross_models/unique_model_counts_boxplot_grouped_by_all_embeddings.png",
                bbox_inches='tight',
                dpi=300)
    plt.close()


def plot_unique_model_counts_per_prompt_vertical():
    data_path = "data/adhoc_save/050925_final_analysis_model_gen/unique_model_counts/unique_model_counts_per_prompt_grouped_by_all_embeddings.json"
    with open(data_path, "r") as f:
        all_unique_model_counts_per_prompt = json.load(f)

    selected_unique_model_counts_per_prompt = {}
    for j in all_unique_model_counts_per_prompt:
        print(
            f"Average number of source models in closest {j} responses: {np.mean(all_unique_model_counts_per_prompt[j]):.2f}")
        j = int(j)
        if (j % 5) == 0 and j > 0:
            selected_unique_model_counts_per_prompt[j] = all_unique_model_counts_per_prompt[str(
                j)]

    # Create box plots for unique model counts
    plt.figure(figsize=(5, 4))

    # Convert data to format for box plot
    plot_data = [selected_unique_model_counts_per_prompt[i]
                 for i in selected_unique_model_counts_per_prompt]

    # (75, 165, 157)
    box_color = (0, 0, 0)
    box_color = [x/255 for x in box_color]
    box_linewidth = 1

    median_color = (237, 110, 87)
    median_color = [x/255 for x in median_color]

    # Create box plot with custom colors
    plt.boxplot(plot_data, vert=False, medianprops=dict(color=median_color, linewidth=1.5),
                boxprops=dict(color=box_color, alpha=1,
                              linewidth=box_linewidth),
                whiskerprops=dict(color=box_color, alpha=1,
                                  linewidth=box_linewidth),
                capprops=dict(color=box_color, alpha=1,
                              linewidth=box_linewidth),
                flierprops=dict(marker='o', markeredgecolor=box_color,
                                markersize=8, alpha=1, linewidth=box_linewidth),
                widths=0.65)

    # Add individual points with jitter
    for i, data in enumerate(plot_data, 1):
        # Create random y-coordinates centered on the box plot position
        y = np.random.normal(i, 0.05, size=len(data))
        plt.scatter(data, y, alpha=0.5, color=(75/255, 165/255, 157/255), s=15)

    plt.title('# Unique Source Models for N Closest\nResponses of Each Open-Ended Query',
              fontsize=11, wrap=True)
    plt.ylabel('N Closest Responses', fontsize=11)
    plt.xlabel('# Unique Source Models', fontsize=11)

    # Set y-axis labels
    plt.yticks(range(1, len(selected_unique_model_counts_per_prompt) + 1),
               [str(j) for j in selected_unique_model_counts_per_prompt], fontsize=10)

    plt.xticks(fontsize=10)
    plt.xlim(1.1, 11.2)

    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig("data/adhoc_save/050925_final_analysis_model_gen/unique_model_counts/unique_model_counts_boxplot_grouped_by_all_embeddings_vertical.png",
                bbox_inches='tight',
                dpi=300)
    plt.close()


if __name__ == "__main__":
    # get_unique_model_counts_per_prompt()
    plot_unique_model_counts_per_prompt_vertical()


