from src.utils.main_utils import load_standard_data, write_standard_data
from tqdm import tqdm
import numpy as np
from src.adhoc.constants import model_names, reasoning_model_names, selected_model_names, min_p_selected_model_names
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


def get_pairwise_similarity(embeddings):
    # Convert to numpy array
    embeddings = np.asarray(embeddings)
    
    # Compute norms for all embeddings at once
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Compute dot products between all pairs
    dot_products = np.dot(embeddings, embeddings.T)
    
    # Compute similarities matrix
    similarities = dot_products / np.outer(norms, norms)
    
    # Extract upper triangle (excluding diagonal) to match original output format
    return similarities[np.triu_indices_from(similarities, k=1)]


def main_model_similarity(model_name, sample_size=None, decoding_method="top_p"):
    model_name = model_name.replace("/", "_")

    if decoding_method == "top_p":
        embeddings_path = f"data/adhoc_save/050725_organize_model_generations/embeddings"
    elif decoding_method == "min_p":
        embeddings_path = f"data/adhoc_save/042725_generate_model_response_min_p/embeddings"
    try:
        data = load_standard_data(f"{embeddings_path}/{model_name}.jsonl", is_print=False)
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return []

    model_avg_similarities = []
    for d in tqdm(data, desc=f"Processing {model_name}"):
        if "embeddings" not in d:
            continue
        d_embeddings = d["embeddings"]
        if sample_size is not None and len(d_embeddings) > sample_size:
            d_embeddings = random.sample(d_embeddings, sample_size)
        
        similarities = get_pairwise_similarity(d_embeddings)
        avg_similarity = np.array(similarities).mean()
        model_avg_similarities.append(avg_similarity)

    return model_avg_similarities


def main_heatmap_transposed(sample_size=None, decoding_method="top_p"):
    """
    Plots a heatmap for within-model similarity. (Figure 3)
    """
    # Set Seaborn style
    sns.set_style("whitegrid")

    all_selected_model_names = selected_model_names
    if decoding_method == "min_p":
        all_selected_model_names = min_p_selected_model_names

    # Store data for all models
    all_model_data = []
    model_names_used = []
    for model_name in all_selected_model_names:
        model_avg_similarities = main_model_similarity(model_name, sample_size=sample_size, decoding_method=decoding_method)

        if len(model_avg_similarities) == 0:
            continue
        
        # Calculate histogram data with fixed range from 0 to 1
        counts, bins = np.histogram(model_avg_similarities, bins=10, range=(0, 1))
        # Convert counts to percentages
        percentages = counts / len(model_avg_similarities) * 100
        
        all_model_data.append(percentages)
        model_names_used.append(model_name.split("/")[-1].replace("Meta-", ""))

    # Create array for heatmap
    heatmap_data = np.array(all_model_data).T  # Transpose the data
    
    # Calculate average for each similarity bucket across models
    avg_column = np.mean(heatmap_data, axis=1, keepdims=True)
    
    # Add average column to heatmap data
    heatmap_data = np.hstack((avg_column, heatmap_data))
    
    # Create labels for y-axis (similarity score buckets)
    bucket_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    # Reverse bucket labels to have high values at top
    bucket_labels = bucket_labels[::-1]
    heatmap_data = np.flipud(heatmap_data)

    # Add 'AVG' to model names at the beginning
    model_names_with_avg = ['Average'] + model_names_used

    # Create heatmap
    if decoding_method == "top_p":
        plt.figure(figsize=(16, 6))
    else:
        plt.figure(figsize=(10, 6))

    ax = sns.heatmap(heatmap_data, 
                     annot=True,  # Show values in cells
                     fmt='.1f',   # Format for the annotations
                     annot_kws={'size': 11},  # Increase size of value labels
                     cmap='YlOrRd',  # Color scheme rocket_r reversed
                     xticklabels=model_names_with_avg,
                     yticklabels=bucket_labels,
                     cbar=False)  # Remove colorbar/legend

    plt.ylabel('Similarity Score Ranges', fontsize=13)
    # plt.xlabel('Models')
    # plt.title('Distribution of Response Similarities Across Models', fontsize=12)

    # Move x-axis to top and rotate labels
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left', fontsize=12)
    # Adjust x-tick positions 1px to the left
    for tick in ax.get_xticklabels():
        tick.set_x(tick.get_position()[0] - 1)  # Shift 1px left
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()

    # Save the figure
    plt.savefig(f"data/adhoc_save/050925_final_analysis_model_gen/within_model/similarity_heatmap_transposed_{decoding_method}.png",
                bbox_inches='tight',
                dpi=300)


if __name__ == "__main__":
    # Plot within-model similarity. (Figure 3)
    main_heatmap_transposed(sample_size=50, decoding_method="top_p")
    main_heatmap_transposed(sample_size=50, decoding_method="min_p")
