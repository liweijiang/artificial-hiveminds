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
import plotly.graph_objects as go
import plotly.express as px
import os


def balanced_clustering(
    embeddings: npt.NDArray[np.float32],
    n_clusters: int,
    items_per_cluster: int
) -> Tuple[List[List[int]], npt.NDArray[np.float32]]:
    """
    Perform k-means clustering on embeddings with a fixed number of items per cluster.

    Args:
        embeddings: Array of shape (n_samples, n_dimensions) containing embeddings
        n_clusters: Number of clusters to create
        items_per_cluster: Number of items each cluster should contain

    Returns:
        Tuple containing:
        - List of lists where each sublist contains indices for items in that cluster
        - Array of cluster centroids
    """
    if len(embeddings) != n_clusters * items_per_cluster:
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) must equal "
            f"n_clusters ({n_clusters}) * items_per_cluster ({items_per_cluster})"
        )

    # Initialize with regular k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # Calculate distances to all centroids for each point
    distances = np.zeros((len(embeddings), n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(embeddings - centroids[i], axis=1)

    # Initialize empty clusters
    clusters = [[] for _ in range(n_clusters)]
    remaining_indices = set(range(len(embeddings)))

    # Assign points to clusters while maintaining size constraint
    while remaining_indices:
        # Find the next best assignment
        min_distance = float('inf')
        best_point = None
        best_cluster = None

        for idx in remaining_indices:
            for cluster_idx in range(n_clusters):
                if len(clusters[cluster_idx]) < items_per_cluster:
                    distance = distances[idx, cluster_idx]
                    if distance < min_distance:
                        min_distance = distance
                        best_point = idx
                        best_cluster = cluster_idx

        # Make the assignment
        clusters[best_cluster].append(best_point)
        remaining_indices.remove(best_point)

    # Update centroids based on final clusters
    for i in range(n_clusters):
        cluster_points = embeddings[clusters[i]]
        centroids[i] = np.mean(cluster_points, axis=0)

    return clusters, centroids


def load_model_data(model_name: str):
    model_name = model_name.replace("/", "_")
    embeddings_path = f"data/adhoc_save/050725_organize_model_generations/embeddings/{model_name}.jsonl"

    try:
        data = load_standard_data(embeddings_path, is_print=False)
        return data
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return []


def run_clustering():
    print("Loading data...")
    start_time = time.time()
    all_data = load_all_data_for_clustering(is_reload=True)
    print(f"Data loaded. Time taken: {time.time() - start_time} seconds")

    # Generate some random embeddings for testing
    np.random.seed(42)
    n_clusters = 25
    items_per_cluster = 50
    n_dimensions = 5

    all_clustering_results = {}
    for p, d in tqdm(all_data.items(), total=len(all_data)):
        embeddings = np.array(d["embeddings"])
        # Perform balanced clustering
        clusters, centroids = balanced_clustering(
            embeddings, n_clusters, items_per_cluster)

        all_clustering_results[p] = {
            "clusters": clusters,
            "centroids": centroids.tolist()  # Convert to list for JSON serialization
        }

        # Visualize clusters
        visualize_clusters(embeddings, clusters, centroids, p)

    with open("data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/all_clustering_results.json", "w") as f:
        json.dump(all_clustering_results, f)


def visualize_clusters(embeddings, clusters, centroids, prompt):
    # Reduce dimensionality to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    centroids_2d = pca.transform(centroids)

    # Create a color palette for the clusters
    n_clusters = len(clusters)
    palette = sns.color_palette("husl", n_clusters)

    # Create the plot
    plt.figure(figsize=(12, 12))

    # Plot points
    for i, cluster in enumerate(clusters):
        cluster_points = embeddings_2d[cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[palette[i]], label=f'Cluster {i}',
                    alpha=0.6)

    # Plot centroids
    # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
    #             c='navy', marker='o', s=200, linewidths=3,
    #             label='Centroids')

    plt.title(f'Cluster Visualization for Prompt:\n{prompt[:100]}...')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    save_path = f"data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/cluster_plots/{prompt[:50].replace(' ', '_')}.png"
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_PCA_by_models(query, all_data):
    d = all_data[query]
    embeddings = np.array(d["embeddings"])
    models = d["models"]

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create a color map for unique models
    unique_models = list(set(models))
    color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_models)))
    model_to_color = dict(zip(unique_models, color_map))

    # Create the scatter plot with high resolution
    plt.figure(figsize=(8, 7.6), dpi=800)
    for model in unique_models:
        mask = [m == model for m in models]
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[model_to_color[model]],
            label=model.split("/")[-1].replace("Meta-", ""),
            alpha=0.7
        )

    plt.title(
        # f'PCA Visualization of Embeddings by Model\nQuery: {query}',
        f'{query}',
        fontsize=14)
    plt.xlabel('First Principal Component', fontsize=14)
    plt.ylabel('Second Principal Component', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=10, frameon=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the plot in high resolution
    save_path = f"data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/PCA/{query[:50].replace(' ', '_')}.png"
    if not os.path.exists(save_path):   
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path,
                bbox_inches='tight',
                dpi=800)
    plt.close()


def load_all_data_responses_for_clustering():
    with open("data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/all_data_responses_for_clustering.json", "r") as f:
        return json.load(f)


def plot_PCA_by_models_interactive(query, all_data, all_data_responses):
    d = all_data[query]
    embeddings = np.array(d["embeddings"])
    models = d["models"]
    responses = all_data_responses[query]

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create a color map for unique models
    unique_models = list(set(models))
    color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_models)))
    model_to_color = dict(zip(unique_models, color_map))
    

    # Create the interactive scatter plot using plotly
    fig = go.Figure()

    for model in unique_models:
        mask = [m == model for m in models]
        mask_indices = np.where(mask)[0]
        fig.add_trace(go.Scatter(
            x=embeddings_2d[mask, 0],
            y=embeddings_2d[mask, 1],
            mode='markers',
            name=model,
            marker=dict(
                color=f'rgba({int(model_to_color[model][0]*255)}, {int(model_to_color[model][1]*255)}, {int(model_to_color[model][2]*255)}, 0.7)'
            ),
            text=[f"Model: {model}<br>Response: {responses[i]}" for i in mask_indices],
            hoverinfo='text',
            customdata=[[model, responses[i]] for i in mask_indices]
        ))

    # Update layout with modal container
    fig.update_layout(
        title=f'PCA Visualization of Embeddings by Model<br>Prompt: {query[:50]}...',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )

    # Create a div for the modal
    modal_html = '''
    <div id="response-modal" class="modal" style="display: none; position: fixed; 
         z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; 
         background-color: rgba(0,0,0,0.4);">
        <div class="modal-content" style="background-color: white; margin: 15% auto; 
             padding: 20px; border: 1px solid #888; width: 80%; max-height: 70%; 
             overflow-y: auto;">
            <span class="close" style="color: #aaa; float: right; font-size: 28px; 
                  font-weight: bold; cursor: pointer;">&times;</span>
            <p id="modal-text"></p>
        </div>
    </div>
    '''

    # Add JavaScript for handling clicks
    click_handler = '''
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var graphDiv = document.querySelector('.js-plotly-plot');
            var modal = document.getElementById('response-modal');
            var modalText = document.getElementById('modal-text');
            var span = document.getElementsByClassName('close')[0];

            graphDiv.on('plotly_click', function(data) {
                var point = data.points[0];
                modalText.innerHTML = '<strong>Model:</strong> ' + point.customdata[0] + 
                                    '<br><br><strong>Response:</strong><br>' + 
                                    point.customdata[1];
                modal.style.display = 'block';
            });

            span.onclick = function() {
                modal.style.display = 'none';
            }

            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }
        });
    </script>
    '''

    # Save the interactive plot as HTML with modal and click handling
    save_path = f"data/adhoc_save/050925_final_analysis_model_gen/clustering/top_p/PCA/{query[:50].replace(' ', '_')}_interactive.html"
    
    with open(save_path, 'w') as f:
        # Write the basic plot
        plot_html = fig.to_html(include_plotlyjs=True, full_html=True)
        
        # Insert modal and click handler before </body>
        plot_html = plot_html.replace('</body>', f'{modal_html}\n{click_handler}\n</body>')
        
        f.write(plot_html)


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


if __name__ == "__main__":
    # run_clustering()

    print("Loading data...")
    start_time = time.time()
    all_data = load_all_data_for_clustering(is_reload=True)
    print(f"Data loaded. Time taken: {time.time() - start_time} seconds")
    all_data_responses = load_all_data_responses_for_clustering()

    # query = "Write a paragraph about how the internet shaped society."
    # query = "Write a metaphor involving time."
    # query = "Create a slogan for a cosmetic bag."
    # query = "Generate a joke about electric vehicles."
    queries = [
        # "Write a metaphor involving time.",
        # "Generate a joke about electric vehicles.",
        # "Write a short story about a colorful toad goes on an adventure in 50 words.",
        # "Write a paragraph about how the internet shaped society.",

        # "Write a pun about peanut.",
        # "Name one meaning of life.",
        # "Write a one- or two-sentence visual description of a game controller.",

        "Provide a few sentences on Sisu Cinema Robotics.",
    ]

    for query in queries:
        plot_PCA_by_models_interactive(query, all_data, all_data_responses)
        plot_PCA_by_models(query, all_data)




