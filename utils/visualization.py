import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from utils.validation import map_results
import networkx as nx

def visualize_dataset(dataset, ground_truth, false_color, filepath):
    print(ground_truth.shape)

    # Figure with better layout and size
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: False Color (RGB) with custom color map
    im0 = ax[0].imshow(false_color)
    ax[0].set_title("False Color (RGB)", fontsize=12)
    ax[0].axis("off")

    # Plot 2: Mean image with normalization and a color bar
    mean_image = np.mean(dataset, axis=2)
    im1 = ax[1].imshow(mean_image, cmap="jet", vmin=np.min(mean_image), vmax=np.max(mean_image))
    ax[1].set_title("Mean Image", fontsize=12)
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)  # Add color bar for mean image

    # Plot 3: Ground Truth with color bar
    im2 = ax[2].imshow(ground_truth, cmap="jet")
    ax[2].set_title("Ground Truth", fontsize=12)
    ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)  # Add color bar for ground truth

    # Tight layout with better spacing
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)  # Keep the high resolution
    # plt.show()


def visualize_segmentation(segments, false_image, ground_truth, filepath):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Increased figure size for better clarity

    # Plot 1: Segmentation boundary with 'jet' colormap
    boundaries = mark_boundaries(false_image, segments)
    ax[0].imshow(boundaries, cmap="jet", vmin=0)  # Using 'jet' colormap for consistency
    ax[0].set_title("Segmentation Boundary", fontsize=14)
    ax[0].axis("off")

    # Calculate the segmentation performance map
    unique_labels = np.unique(segments)
    seg_map = np.zeros_like(segments)

    for label in unique_labels:
        mask = segments == label
        counts = np.bincount(ground_truth[mask])
        if counts.size > 1 and np.any(counts[1:] > 0):
            seg_map[mask] = np.argmax(counts[1:]) + 1
        else:
            seg_map[mask] = 0

    # Calculate accuracy metrics
    oa, aa, ka, _, _ = map_results(seg_map, ground_truth)

    # Print the results with formatted output
    print(f'OA: {oa:.4f}, AA: {aa:.4f}, KA: {ka:.4f}')

    # Plot 2: Segmentation performance map with color bar
    im1 = ax[1].imshow(seg_map, cmap="jet", vmin=np.min(seg_map), vmax=np.max(seg_map))
    ax[1].set_title(f"Segmentation Performance, OA: {oa:.4f}", fontsize=14)
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)  # Add color bar for segmentation performance

    # Save the visualization
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)  # Save with high DPI for clarity
    # plt.show()


def visualize_cmap(cmap, ground_truth, filepath):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Larger figure for better clarity

    # Plot 1: Original output with a color bar
    im0 = ax[0].imshow(cmap, cmap="jet", vmin=0, vmax=np.max(cmap))
    ax[0].set_title("Model Output", fontsize=14)
    ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # Plot 2: Output clipped by ground truth, using masked array
    cmap_clipped = cmap
    cmap[ground_truth == 0] = 0
    im1 = ax[1].imshow(cmap_clipped, cmap="jet", vmin=0, vmax=np.max(cmap))
    ax[1].set_title("Output with Ground Truth Mask", fontsize=14)
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # Improve layout and save the figure
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)  # Save with high resolution
    # plt.show()


def visualize_graph(graph, filepath):
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    plt.figure(figsize=(12, 8))  # Larger figure for better visibility

    # Use a spring layout for better aesthetics
    pos = nx.spring_layout(graph)

    # Draw the graph with enhanced aesthetics
    nx.draw(graph, pos, node_size=50, edge_color='gray', alpha=0.5,
            node_color='skyblue', with_labels=False)  # Adjust colors and sizes as needed

    # Optionally, draw node labels (comment out if not needed)
    # nx.draw_networkx_labels(graph, pos, font_size=8)

    plt.title('Graph Representation of Superpixel Segmentation', fontsize=16)
    plt.axis('off')  # Turn off the axis for a cleaner look
    plt.savefig(filepath, dpi=600)  # Save with high DPI for clarity
    # plt.show()


def plot_training_results(EPOCH, loss_history, accuracy_history, filepath):
  # Create a figure with two subplots
  fig, axs = plt.subplots(1, 2, figsize=(14, 6))

  # Plot loss history
  axs[0].plot(range(1, EPOCH + 2), loss_history, 'b', label='Training loss')
  axs[0].set_title('Training Loss')
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Loss')
  axs[0].legend()

  # Plot accuracy history
  axs[1].plot(range(1, EPOCH + 2), accuracy_history, 'r', label='Training accuracy')
  axs[1].set_title('Training Accuracy')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Accuracy')
  axs[1].legend()

  plt.tight_layout()
  plt.savefig(filepath, dpi=600)
  # plt.show()


def visualize_cmap_compare_ground_truth(cmap, ground_truth, filepath):
  fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Larger figure for better clarity

  # Plot 1: Original output with a color bar
  cmap_clipped = cmap.copy()
  cmap_clipped[ground_truth == 0] = 0

  im0 = ax[0].imshow(cmap_clipped, cmap="jet", vmin=0, vmax=np.max(cmap))
  ax[0].set_title("Model Output", fontsize=14)
  ax[0].axis("off")
  plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

  # Plot 2: Ground Truth
  im1 = ax[1].imshow(ground_truth, cmap="jet", vmin=0, vmax=np.max(ground_truth))
  ax[1].set_title("Ground Truth", fontsize=14)
  ax[1].axis("off")
  plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

  # Improve layout and save the figure
  plt.tight_layout()
  plt.savefig(filepath, dpi=600)  # Save with high resolution
  # plt.show()