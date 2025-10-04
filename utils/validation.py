import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix
)

def map_results(seg_map, ground_truth):
    # Flatten the input arrays
    seg_map = np.array(seg_map).flatten()
    ground_truth = np.array(ground_truth).flatten()

    # Filter out zero indices from ground_truth
    non_zero_indices = np.where(ground_truth != 0)
    filtered_ground_truth = ground_truth[non_zero_indices]
    filtered_seg_map = seg_map[non_zero_indices]

    # Calculate overall accuracy
    overall_acc = accuracy_score(filtered_ground_truth, filtered_seg_map)

    # Calculate average (balanced) accuracy
    class_acc = balanced_accuracy_score(filtered_ground_truth, filtered_seg_map)

    # Calculate Cohen's Kappa score
    kappa_score = cohen_kappa_score(filtered_ground_truth, filtered_seg_map)

    # Generate classification report, excluding class 0
    report = classification_report(
        filtered_ground_truth,
        filtered_seg_map,
        labels=np.unique(filtered_ground_truth),  # Use unique labels from ground truth
        zero_division=0  # Handle zero division
    )

    # Compute confusion matrix, excluding class 0
    matrix = confusion_matrix(
        filtered_ground_truth,
        filtered_seg_map,
        labels=np.unique(filtered_ground_truth)  # Exclude class 0
    )

    return overall_acc, class_acc, kappa_score, report, matrix