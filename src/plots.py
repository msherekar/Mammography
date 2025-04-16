import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, confusion_matrix, roc_auc_score
import seaborn as sns
import numpy as np
import os
import pandas as pd

def plot_training_metrics(epochs, losses, aucs_training, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color='blue', marker='o', markeredgewidth=2, linestyle='-', label='Training Loss')
    plt.plot(epochs, aucs_training, color='red', marker='o', markeredgewidth=2, linestyle='-', label='Tuning AUC')
    plt.xlabel('Epoch', fontsize=12)
    plt.title('Training Loss & Tuning AUC Across Epochs', fontsize=14, fontweight='bold')
    plt.xticks(epochs)  # Ensure that the x-axis shows the epochs as integers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper right', fontsize=10, frameon=False)
    plt.savefig(os.path.join(out_dir, f'Loss_Curve_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_sensitivity_specificity_auc(epochs, sensitivities, specificities, aucs, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sensitivities, color='red', marker='o', markeredgewidth=2, linestyle='-', label='Sensitivity')
    plt.plot(epochs, specificities, color='green', marker='x', markeredgewidth=2, linestyle='-', label='Specificity')
    plt.plot(epochs, aucs, color='black', marker='.', markeredgewidth=2, linestyle='-', label='AUC')
    plt.xlabel('Epoch', fontsize=12)
    plt.title('Sensitivity, Specificity, and AUC Across Epochs', fontsize=14, fontweight='bold')
    plt.xticks(epochs)  # Ensure that the x-axis shows the epochs as integers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center right', fontsize=10, frameon=False)
    plt.savefig(os.path.join(out_dir, f'Sens_Spec_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_csaw_metrics(epochs, sensitivities_mcp, specificities_mcn, aucs, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sensitivities_mcp, color='red', marker='o', markeredgewidth=2, linestyle='-', label='Sensitivity Cancer +ve')
    plt.plot(epochs, specificities_mcn, color='green', marker='x', markeredgewidth=2, linestyle='--', label='Specificity Cancer -ve')
    plt.plot(epochs, aucs, color='black', marker='.', markeredgewidth=2, linestyle='-', label='AUC')
    plt.xlabel('Epoch', fontsize=12)
    plt.title('Sensitivity & Specificity of csaw Images Across Epochs', fontsize=14, fontweight='bold')
    plt.xticks(epochs)  # Ensure that the x-axis shows the epochs as integers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center right', fontsize=10, frameon=False)
    plt.savefig(os.path.join(out_dir, f'Metrics_csaw_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_embed_metrics(epochs, sensitivities_ecp, specificities_ecn, aucs, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, sensitivities_ecp, color='red', marker='o', markeredgewidth=2, linestyle='-', label='Sensitivity Cancer +ve')
    plt.plot(epochs, specificities_ecn, color='green', marker='x', markeredgewidth=2, linestyle='--', label='Specificity Cancer -ve')
    plt.plot(epochs, aucs, color='black', marker='.', markeredgewidth=2, linestyle='-', label='AUC')
    plt.xlabel('Epoch', fontsize=12)
    plt.title('Sensitivity & Specificity of EMBED Images Across Epochs', fontsize=14, fontweight='bold')
    plt.xticks(epochs)  # Ensure that the x-axis shows the epochs as integers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='center right', fontsize=10, frameon=False)
    plt.savefig(os.path.join(out_dir, f'Metrics_EMBED_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_roc_curve(fpr,tpr,auc_val,fpr_csaw,tpr_csaw,auc_csaw,fpr_embed,tpr_embed,auc_embed,out_dir,exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='Combined AUC = %.2f' % auc_val, color='black')
    plt.plot(fpr_csaw, tpr_csaw, label='CSAW AUC = %.2f' % auc_csaw, color='blue')
    plt.plot(fpr_embed, tpr_embed, label='EMBED AUC = %.2f' % auc_embed, color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='green')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves')
    plt.legend(loc='lower right', prop={'size': 12, 'weight': 'bold'})
    plt.savefig(os.path.join(out_dir, f'ROC_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_csaw_roc_curve(fpr,tpr,auc_val,out_dir,exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='AUC = %.2f' % auc_val, color='black')
    plt.plot([0, 1], [0, 1], linestyle='--', color='green')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend(loc='lower right', prop={'size': 12, 'weight': 'bold'})
    plt.savefig(os.path.join(out_dir, f'ROC_{exp_name}.png'))
    plt.show()
    plt.close()
def plot_score_distribution(result_df1, out_dir, exp_name):
    """
    Plots the score distribution based on the result dataframe and saves the plot.
    Parameters:
    result_df1 (pd.DataFrame): DataFrame containing 'score' and 'true_labels' columns.
    out_dir (str): Directory where the plot will be saved.
    exp_name (str): Experiment name to include in the saved plot filename.
    """

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    xmax = result_df1["score"].max()

    # Range for bins
    low_percentile = result_df1['score'].quantile(0.01)
    high_percentile = result_df1['score'].quantile(0.99)

    # Create the FacetGrid for plotting
    g = sns.FacetGrid(data=result_df1, hue="true_labels", height=6, palette={0: 'blue', 1: 'red'})
    g.map(sns.histplot, "score", bins=30, kde=True, binrange=(low_percentile, high_percentile), stat='count', alpha=0.5, element="step")
    #sns.histplot(data=result_df1, x='score', hue='true_labels', palette={0: 'blue', 1: 'red'}, kde=True, stat='count', element="step")

    # Set plot attributes
    plt.title('Distribution of Predicted Probabilities for Images in Tuning Set')
    plt.xlabel('Probabilities')
    plt.ylabel('Counts')
    plt.xlim(0, xmax)
    
    plt.legend(title='True Labels', labels=['Class 0 (Blue)', 'Class 1 (Red)'], loc='upper right')

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'PredProb_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_score_distribution_datasets(result_df1, out_dir, exp_name):
    """
    Plots the score distribution for different datasets ('csaw' and 'embed') and saves the plot.
    Parameters:
    result_df1 (pd.DataFrame): DataFrame containing 'score', 'true_labels', and 'dataset' columns.
    out_dir (str): Directory where the plot will be saved.
    exp_name (str): Experiment name to include in the saved plot filename.
    """
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 8))
    
    # Range for bins
    low_percentile = result_df1['score'].quantile(0.01)
    high_percentile = result_df1['score'].quantile(0.99)
    
    # Define xmax for the x-axis limit
    xmax = result_df1["score"].max()
    
    # Create combined 'label' column for dataset and true_labels
    result_df1['label'] = result_df1['dataset'].astype(str) + ' - ' + result_df1['true_labels'].astype(str)
    
    # Generate dynamic color palette
    unique_labels = result_df1['label'].unique()
    color_palette = sns.color_palette("tab10", len(unique_labels))
    palette_dict = dict(zip(unique_labels, color_palette))
    
    # Create the plot
    ax = sns.histplot(data=result_df1, x='score', hue='label', palette=palette_dict, kde=True, binrange=(low_percentile, high_percentile),stat='count', element="step")
    
    # Set titles and labels
    plt.title('Distribution of Predicted Probabilities Based on Datasets for Images in Tuning Set')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Counts')
    plt.xlim(0, xmax)
    plt.legend(title='Dataset - Label', loc='upper right')
    
    # Create custom legend labels that match the colors
    legend_labels = [f'{key} ({value})' for key, value in color_palette.items()]
    ax.legend(title='Dataset Type - Label (Color)', labels=legend_labels)
    
    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Score_dataset_{exp_name}.png'))
    
    # Final adjustments and display
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_uncertainty_distribution(result_df1, out_dir, exp_name):
    """
    Plots the uncertainty distribution based on predicted labels and saves the plot.

    Parameters:
    result_df1 (pd.DataFrame): DataFrame containing 'uncertainty' and 'predicted_labels' columns.
    out_dir (str): Directory where the plot will be saved.
    exp_name (str): Experiment name to include in the saved plot filename.
    """

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # Plot uncertainty distribution
    sns.histplot(data=result_df1, x='uncertainty', hue='predicted_labels', palette={0: 'blue', 1: 'red'}, kde=True, element="step")

    # Set plot titles and labels
    plt.title('Uncertainty Distribution for Predicted Labels')
    plt.xlabel('Uncertainty')
    plt.ylabel('Count')

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Uncertainty_{exp_name}.png'))

    # Show and close the plot
    plt.show()
    plt.close()

def plot_logits_scatter(result_df1, out_dir, exp_name):
    """
    Plots a scatter plot of logits with colors representing different dataset types and labels.

    Parameters:
    result_df1 (pd.DataFrame): DataFrame containing 'logits', 'true_labels', and 'dataset' columns.
    out_dir (str): Directory where the plot will be saved.
    exp_name (str): Experiment name to include in the saved plot filename.
    """

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # Define color palette based on dataset type and true labels
    color_map = {'embed - 0': 'green', 'embed - 1': 'red', 'csaw - 0': 'black', 'csaw - 1': 'blue'}

    # Create a new column 'label' combining dataset type and true labels
    result_df1['label'] = result_df1['dataset'].astype(str) + ' - ' + result_df1['true_labels'].astype(str)

    # Scatter plot: iterate over the unique labels and plot each group separately
    for label, color in color_map.items():
        subset = result_df1[result_df1['label'] == label]
        plt.scatter(range(len(subset)), subset['logits'], color=color, s=1, label=label, alpha=0.6)

    # Set plot labels and title
    plt.title('Logits Scores for Tuning Set')
    plt.xlabel('Image index in CSV file')
    plt.ylabel('Logits Score')

    # Show grid
    plt.grid(True)

    # Add a legend with customized markers
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                           markerfacecolor=color, markersize=10, markeredgecolor='black', linewidth=2) 
               for label, color in color_map.items()]
    
    plt.legend(title='Dataset Type and True Labels', handles=handles, prop={'weight': 'bold', 'size': 14})


    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logits_scatter_{exp_name}.png'))

    # Show and close the plot
    plt.show()
    plt.close()

def plot_logits_distribution(logits, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # Define bins from minimum to maximum logits with a width of 0.5
    min_logit = np.floor(np.min(logits))  # Find the minimum logit score
    max_logit = np.ceil(np.max(logits))   # Find the maximum logit score
    bins = np.arange(min_logit, max_logit + 0.5, 0.5)  # Create bins with a width of 0.5

    # Plot the distribution of logits scores using a histogram with raw counts
    sns.histplot(logits, bins=bins, color='green', stat='count', label='CSAW', alpha=0.6)

    plt.title('Logit Scores Distribution', fontsize=14)
    plt.xlabel('Logit Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logits_Distribution_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_logits_distribution_by_confusion_matrix(logits, true_labels, out_dir, exp_name, threshold):
    """
    Plot logits distribution categorized by confusion matrix outcomes (TP, FP, TN, FN).
    """
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 8))
    
    # Convert inputs to numpy arrays
    true_labels = np.array(true_labels)
    logits = np.array(logits)
    
    # Create masks for each category
    TP = (true_labels == 1) & (logits > threshold)
    FP = (true_labels == 0) & (logits > threshold)
    TN = (true_labels == 0) & (logits <= threshold)
    FN = (true_labels == 1) & (logits <= threshold)
    
    # Calculate summary statistics
    total_samples = len(logits)
    accuracy = (np.sum(TP) + np.sum(TN)) / total_samples
    precision = np.sum(TP) / (np.sum(TP) + np.sum(FP)) if (np.sum(TP) + np.sum(FP)) > 0 else 0
    recall = np.sum(TP) / (np.sum(TP) + np.sum(FN)) if (np.sum(TP) + np.sum(FN)) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Define colors and labels with better contrast
    categories = {
        'TP': {'color': '#2ecc71', 'label': 'True Positives', 'alpha': 0.7},
        'FP': {'color': '#e74c3c', 'label': 'False Positives', 'alpha': 0.7},
        'TN': {'color': '#3498db', 'label': 'True Negatives', 'alpha': 0.7},
        'FN': {'color': '#f39c12', 'label': 'False Negatives', 'alpha': 0.7}
    }
    
    # Calculate bin edges with better spacing
    min_logit = np.floor(np.min(logits) * 2) / 2  # Round to nearest 0.5
    max_logit = np.ceil(np.max(logits) * 2) / 2   # Round to nearest 0.5
    bin_width = 0.5
    bins = np.arange(min_logit, max_logit + bin_width, bin_width)
    
    # Function to add value labels on bars
    def add_value_labels(values, category):
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        
        for count, center in zip(counts, centers):
            if count > 0:
                plt.text(center, count, str(int(count)),
                        ha='center', va='bottom',
                        color=categories[category]['color'],
                        fontweight='bold', fontsize=8)
    
    # Plot histograms
    for category, mask in zip(['TP', 'FP', 'TN', 'FN'], [TP, FP, TN, FN]):
        plt.hist(logits[mask], bins=bins,
                color=categories[category]['color'],
                alpha=categories[category]['alpha'],
                label=f"{categories[category]['label']} (n={np.sum(mask)})",
                edgecolor='black',
                linewidth=0.5)
        add_value_labels(logits[mask], category)
    
    # Add threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.5,
                label=f'Threshold = {threshold:.2f}')
    
    # Customize axes
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1.0))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add y-axis grid only
    plt.grid(axis='y', alpha=0.3)
    
    # Add summary statistics text box
    stats_text = (

        f'Class Distribution:\n'
        f'TP: {np.sum(TP)} ({np.sum(TP)/total_samples:.1%})\n'
        f'FP: {np.sum(FP)} ({np.sum(FP)/total_samples:.1%})\n'
        f'TN: {np.sum(TN)} ({np.sum(TN)/total_samples:.1%})\n'
        f'FN: {np.sum(FN)} ({np.sum(FN)/total_samples:.1%})'
    )
    
    plt.text(1.25, 0.97, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace')
    
    # Add labels and title
    plt.title('Distribution of Logit Scores by Classification Outcome',
              fontsize=14, pad=20)
    plt.xlabel('Logit Score', fontsize=12, labelpad=10)
    plt.ylabel('Count', fontsize=12, labelpad=10)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left',
              borderaxespad=0., frameon=True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'Logits_cf_{exp_name}.png'),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_logits_distribution_by_dataset(logits, dataset_types, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # Define bins from minimum to maximum logits with a width of 0.5
    min_logit = np.floor(np.min(logits))
    max_logit = np.ceil(np.max(logits))
    bins = np.arange(min_logit, max_logit + 0.5, 0.5)

    # Create a color palette for dataset types
    color_palette = {'embed': 'green', 'csaw': 'blue'}

    # Initialize counts for each dataset type
    counts = {dataset_type: np.zeros(len(bins) - 1) for dataset_type in color_palette.keys()}

    # Count occurrences in each bin based on dataset type
    for i, logit in enumerate(logits):
        dataset_type = dataset_types[i]
        if dataset_type in counts:
            counts[dataset_type] += np.histogram([logit], bins=bins)[0]

    # Create an offset for the bars
    bar_width = 0.4  # Width of each bar
    x = np.arange(len(bins) - 1)  # Bin positions

    # Plot bars for each dataset type
    plt.bar(x - bar_width / 2, counts['embed'], width=bar_width, color=color_palette['embed'], label='EMBED', alpha=0.6)
    plt.bar(x + bar_width / 2, counts['csaw'], width=bar_width, color=color_palette['csaw'], label='CSAW', alpha=0.6)

    plt.title('Logits Distribution Colored by Dataset Type', fontsize=14)
    plt.xlabel('Logit Score Bins', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)], rotation=45)  # Set x-axis ticks to show bin ranges
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logits_Distribution_By_Dataset_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_logit_counts_grid(logits, out_dir, exp_name):
    plt.style.use('seaborn-darkgrid')

    # Define bins from minimum to maximum logits with a width of 0.5
    min_logit = np.floor(np.min(logits))
    max_logit = np.ceil(np.max(logits))
    bins = np.arange(min_logit, max_logit + 0.5, 0.5)

    # Count occurrences in each bin
    counts, _ = np.histogram(logits, bins=bins)

    # Reshape counts into a grid (2D array)
    # You can customize the grid shape; here we assume a 1D array to fit in a 2D grid
    grid_size = int(np.ceil(np.sqrt(len(counts))))  # Determine grid size based on count length
    grid_counts = np.zeros((grid_size, grid_size))
    
    # Fill grid counts
    for i in range(len(counts)):
        row = i // grid_size
        col = i % grid_size
        grid_counts[row, col] = counts[i]

    # Create a heatmap for visualization
    plt.figure(figsize=(8, 8))
    sns.heatmap(grid_counts, annot=True, fmt='.0f', cmap='coolwarm', cbar=True, square=True,linewidths=0.5, linecolor='black', vmin=0, vmax=np.max(counts))

    plt.title('Logit Score Counts Grid', fontsize=14)
    plt.xlabel('Bins', fontsize=12)
    plt.ylabel('Counts', fontsize=12)

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logit_Score_Counts_Grid_{exp_name}.png'))
    plt.show()
    plt.close()

# Function to calculate TP, FP, TN, FN counts and sensitivity, specificity
def calculate_metrics(df, dataset_value):
    # Filter by dataset
    dataset_df = df[df['dataset'] == dataset_value]
    
    # True Positives (TP): predicted_labels == 1 and true_labels == 1
    TP = ((dataset_df['predicted_labels'] == 1) & (dataset_df['true_labels'] == 1)).sum()
    
    # True Negatives (TN): predicted_labels == 0 and true_labels == 0
    TN = ((dataset_df['predicted_labels'] == 0) & (dataset_df['true_labels'] == 0)).sum()
    
    # False Positives (FP): predicted_labels == 1 and true_labels == 0
    FP = ((dataset_df['predicted_labels'] == 1) & (dataset_df['true_labels'] == 0)).sum()
    
    # False Negatives (FN): predicted_labels == 0 and true_labels == 1
    FN = ((dataset_df['predicted_labels'] == 0) & (dataset_df['true_labels'] == 1)).sum()
    
    # Sensitivity (True Positive Rate): TP / (TP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Specificity (True Negative Rate): TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return TP, FP, TN, FN, sensitivity, specificity


def save_last_epoch_metrics(folder_name, file_name):
    """
    Save metrics from the last epoch's results file in the specified folder.

    Parameters:
        folder_name (str): The folder containing the result `.tsv` files.
        file_name (str): The name of the output CSV file to save the metrics.
    """
    # Ensure the output folder exists
    output_file = os.path.join(folder_name, f'{file_name}_metrics.csv')
    results = []

    # Search for the last epoch result `.tsv` file
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if file.startswith('results') and file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                
                # Read the .tsv file into a DataFrame
                df = pd.read_csv(file_path, sep='\t')
                
                # Calculate metrics for 'embed' dataset rows
                embed_TP, embed_FP, embed_TN, embed_FN, embed_sensitivity, embed_specificity = calculate_metrics(df, 'embed')
                
                # Calculate metrics for 'csaw' dataset rows
                csaw_TP, csaw_FP, csaw_TN, csaw_FN, csaw_sensitivity, csaw_specificity = calculate_metrics(df, 'csaw')

                # Calculate combined metrics for both datasets
                TP = embed_TP + csaw_TP
                FP = embed_FP + csaw_FP
                TN = embed_TN + csaw_TN
                FN = embed_FN + csaw_FN
                combined_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
                combined_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                
                # Append results to the list
                results.append({
                    'file_name': file, 
                    'embed_TP': embed_TP,
                    'embed_FP': embed_FP,
                    'embed_TN': embed_TN,
                    'embed_FN': embed_FN,
                    'csaw_TP': csaw_TP,
                    'csaw_FP': csaw_FP,
                    'csaw_TN': csaw_TN,
                    'csaw_FN': csaw_FN,
                    'combined_TP': TP,
                    'combined_FP': FP,
                    'combined_TN': TN,
                    'combined_FN': FN,
                    'embed_sensitivity': embed_sensitivity,
                    'embed_specificity': embed_specificity,
                    'csaw_sensitivity': csaw_sensitivity,
                    'csaw_specificity': csaw_specificity,
                    'combined_sensitivity': combined_sensitivity,
                    'combined_specificity': combined_specificity
                })

    # Convert results to DataFrame and export to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    #print(f"Metrics for the last epoch saved to {output_file}")

def plot_logits_distribution_by_true_labels(logits, true_labels, out_dir, exp_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 6))

    # Define bins from minimum to maximum logits with a width of 0.5
    min_logit = np.floor(np.min(logits))
    max_logit = np.ceil(np.max(logits))
    bins = np.arange(min_logit, max_logit + 0.5, 0.5)

    # Initialize counts for each true label
    counts = {0: np.zeros(len(bins) - 1), 1: np.zeros(len(bins) - 1)}

    # Count occurrences in each bin based on true label
    for i, logit in enumerate(logits):
        true_label = true_labels[i]
        if true_label in counts:
            counts[true_label] += np.histogram([logit], bins=bins)[0]

    # Create an offset for the bars
    bar_width = 0.4  # Width of each bar
    x = np.arange(len(bins) - 1)  # Bin positions

    # Plot bars for each true label
    plt.bar(x - bar_width / 2, counts[0], width=bar_width, color='red', label='Non Diseased: 0', alpha=0.6)
    plt.bar(x + bar_width / 2, counts[1], width=bar_width, color='blue', label='Diseased: 1', alpha=0.6)

    plt.title('Logits Distribution by True Labels', fontsize=14)
    plt.xlabel('Logit Score Bins', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)], rotation=45)  # Set x-axis ticks to show bin ranges
    plt.legend(loc='upper right')

    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logits_Distribution_By_True_Labels_{exp_name}.png'))
    plt.show()
    plt.close()

def plot_logits_distribution_by_dataset2(result_df1, out_dir, exp_name):
    """
    Plots the logits distribution for different datasets ('csaw' and 'embed') and saves the plot.
    Parameters:
    result_df1 (pd.DataFrame): DataFrame containing 'logits', 'true_labels', and 'dataset' columns.
    out_dir (str): Directory where the plot will be saved.
    exp_name (str): Experiment name to include in the saved plot filename.
    """
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(12, 8))
    
    # Range for bins
    low_percentile = result_df1['logits'].quantile(0.01)
    high_percentile = result_df1['logits'].quantile(0.99)
    
    # Define xmax for the x-axis limit
    xmax = result_df1["logits"].max()
    
    # Create combined 'label' column for dataset and true_labels
    result_df1['label'] = result_df1['dataset'].astype(str) + ' - ' + result_df1['true_labels'].astype(str)
    
    # Define color palette for different dataset and label combinations
    color_palette = {'csaw - 0': 'blue','csaw - 1': 'red','embed - 0': 'green','embed - 1': 'black'}
    
    # Create the plot
    ax = sns.histplot(data=result_df1, x='logits', hue='label', palette=color_palette, kde=True, stat='count', element="step")
    
    # Set titles and labels
    plt.title('Distribution of logits in Tuning Set')
    plt.xlabel('Logits')
    plt.ylabel('Counts')
    plt.xlim(0, xmax)
    
    # Create custom legend labels that match the colors
    legend_labels = [f'{key} ({value})' for key, value in color_palette.items()]
    ax.legend(title='Dataset Type - Label (Color)', labels=legend_labels)
    
    # Save the plot
    plt.savefig(os.path.join(out_dir, f'Logits2_dataset_{exp_name}.png'))
    
    # Final adjustments and display
    plt.tight_layout()
    plt.show()
    plt.close()