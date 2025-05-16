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


def get_pairwise_similarity_between_two_sets(embeddings_1, embeddings_2, sample_size=10):
    # Convert to numpy arrays
    embeddings_1 = np.asarray(embeddings_1)
    embeddings_2 = np.asarray(embeddings_2)
    
    if sample_size is not None:
        if len(embeddings_1) > sample_size:
            indices = np.random.choice(len(embeddings_1), sample_size, replace=False)
            embeddings_1 = embeddings_1[indices]
        if len(embeddings_2) > sample_size:
            indices = np.random.choice(len(embeddings_2), sample_size, replace=False)
            embeddings_2 = embeddings_2[indices]
    
    # Compute norms for all embeddings at once
    norms_1 = np.linalg.norm(embeddings_1, axis=1)
    norms_2 = np.linalg.norm(embeddings_2, axis=1)
    
    # Compute all similarities at once using matrix multiplication
    similarities = np.dot(embeddings_1, embeddings_2.T) / np.outer(norms_1, norms_2)
    
    # Flatten to match original output format
    return similarities.flatten()


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


def load_model_data(model_name: str, decoding_method="top_p"):
    model_name = model_name.replace("/", "_")

    if decoding_method == "top_p":
        embeddings_path = f"data/adhoc_save/050725_organize_model_generations/embeddings"
    elif decoding_method == "min_p":
        embeddings_path = f"data/adhoc_save/042725_generate_model_response_min_p/embeddings"

    try:
        data = load_standard_data(embeddings_path + f"/{model_name}.jsonl", is_print=False)
        return data
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return []


def main_heatmap_across_models(decoding_method="top_p"):
    """
    Plots a heatmap of the pairwise embedding similarity between models. (Figure 4)
    """
    # Dictionary to store average similarities for each model

    if decoding_method == "min_p":
        from src.adhoc.constants import min_p_selected_model_names as selected_model_names
    else:
        from src.adhoc.constants import selected_model_names

    model_similarities = {}
    for model_name in selected_model_names:
        model_similarities[model_name] = {}

    # Calculate average embeddings for each model
    model_data_cache = {}
    for model_name in tqdm(selected_model_names, desc="Loading model data"):
        model_data_cache[model_name] = load_model_data(model_name, decoding_method)

    for i, model_name in enumerate(tqdm(selected_model_names, desc="Calculating similarities")):
        model_data = model_data_cache[model_name]
        # Only calculate for lower triangle
        for j, model_name_2 in enumerate(selected_model_names[:i+1]):
            model_data_2 = model_data_cache[model_name_2]

            all_similarities = []
            for d, d_2 in zip(model_data, model_data_2):
                if "embeddings" not in d or "embeddings" not in d_2:
                    continue
                similarities = get_pairwise_similarity_between_two_sets(d["embeddings"], d_2["embeddings"], sample_size=None)
                all_similarities.extend(similarities)
            model_similarities[model_name][model_name_2] = np.mean(all_similarities)

    similarity_matrix = np.zeros((len(selected_model_names), len(selected_model_names)))
    # Fill only lower triangle
    for i, model_name in enumerate(selected_model_names):
        for j, model_name_2 in enumerate(selected_model_names[:i+1]):
            similarity_matrix[i,j] = model_similarities[model_name][model_name_2]
            
    # Create heatmap
    plt.figure(figsize=(12, 12))
    mask = np.triu(np.ones_like(similarity_matrix), k=1)
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=[model_name.split("/")[-1].replace("Meta-", "") for model_name in selected_model_names],
        yticklabels=[model_name.split("/")[-1].replace("Meta-", "") for model_name in selected_model_names],
        square=True,
        mask=mask,
        vmin=0.5,  # Minimum value for the colormap scale
        vmax=1,  # Maximum value for the colormap scale 
        cbar=False  # Don't show the colorbar
    )
    
    # plt.title('Pairwise Model Response Similarities')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Format annotations to remove leading zeros
    for t in plt.gca().texts:
        text = t.get_text()
        if text.startswith('0'):
            t.set_text(text[1:])
    
    plt.tight_layout()
    plt.savefig(f"data/adhoc_save/050925_final_analysis_model_gen/cross_models/model_pairwise_similarity_{decoding_method}.png",
                bbox_inches='tight',
                dpi=300)
    plt.close()


if __name__ == "__main__":
    # Plot pairwise similarity between models. (Figure 4)
    main_heatmap_across_models(decoding_method="top_p")
    # main_heatmap_across_models(decoding_method="min_p")
