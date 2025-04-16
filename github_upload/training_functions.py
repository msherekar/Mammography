from imports import *



def count_and_plot_batch_data(loader, output_dir, log_filename='batch_counts.log', plot_filename='batch_counts_plot.png'):
    """
    Counts cancer-positive and cancer-negative images in each batch, saves the results to a log file, 
    and creates a bar chart plot.

    Args:
        loader (DataLoader): The DataLoader object for the dataset.
        output_dir (str): Directory to save the log file and plot.
        log_filename (str): Name of the log file. Default is 'batch_counts.log'.
        plot_filename (str): Name of the plot file. Default is 'batch_counts_plot.png'.
    """
    # Initialize lists for cancer+ve and cancer-ve counts
    batch_indices = []
    cancer_pos_counts = []
    cancer_neg_counts = []

    # Prepare log file path
    log_file = os.path.join(output_dir, log_filename)

    # Open the log file and write the header
    with open(log_file, 'w') as log_fp:
        log_fp.write("Batch\tCancer+ve\tCancer-ve\n")  # Header

    # Iterate through batches
    for batch_idx, (images, labels, _, _) in enumerate(loader):
        cancer_pos = (labels == 1).sum().item()  # Count of cancer+ve
        cancer_neg = (labels == 0).sum().item()  # Count of cancer-ve

        # Append counts for plotting
        batch_indices.append(batch_idx)
        cancer_pos_counts.append(cancer_pos)
        cancer_neg_counts.append(cancer_neg)

        # Log counts to file
        with open(log_file, 'a') as log_fp:
            log_fp.write(f"{batch_idx}\t{cancer_pos}\t{cancer_neg}\n")

    # Plot cancer+ve and cancer-ve counts per batch
    plt.figure(figsize=(12, 6))
    plt.scatter(batch_indices, cancer_pos_counts, label='Cancer+ve', color='red', alpha=0.6)
    plt.scatter(batch_indices, cancer_neg_counts, label='Cancer-ve', color='blue', alpha=0.6)
    plt.xlabel('Batch Index')
    plt.ylabel('Count')
    plt.title('Cancer+ve and Cancer-ve Counts per Batch')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()

    
def confusion_matrix_subset(subset): # subset is a dictionary
    tp = subset['tp']
    fp = subset['fp']
    tn = subset['tn']
    fn = subset['fn']
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return sensitivity, specificity, precision

def calculate_loss_accuracy(subset):
    if subset['total'] == 0:
        return 0, 0  # Avoid division by zero
    avg_loss = subset['loss'] / subset['total']
    avg_accuracy = subset['correct'] / subset['total']
    return avg_loss, avg_accuracy
def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_count = len(labels)
    class_weights = total_count / (len(class_counts) * class_counts)
    sample_weights = np.array([class_weights[label] for label in labels])
    return sample_weights
def append_dropout(model, rate):
    """ Function to add dropout layer after each convolutional layer"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)
        if isinstance(module, torch.nn.Conv2d):
            new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def apply_custom_transfer_learning(model, num_to_freeze):
    # Get all parameter names
    model_layers = [name for name, para in model.named_parameters()]
    
    # Generalized layer grouping approach
    def group_layers(total_layers, num_groups=42):
        if total_layers < num_groups:
            return [','.join(model_layers)]
        
        layers = []
        group_size = max(1, total_layers // num_groups)
        
        for i in range(num_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < num_groups - 1 else total_layers
            layers.append(','.join(model_layers[start:end]))
        
        return layers

    # Generate layers
    layers = group_layers(len(model_layers))
    
    # Select layers to fine-tune
    fine_tune_layers = ','.join(layers[num_to_freeze-len(layers):]).split(',')
    
    # Freeze/Unfreeze parameters
    for name, param in model.named_parameters():
        if name not in fine_tune_layers:
            param.requires_grad = False
    
    # Verify parameters
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    print(f'Total parameters: {len(model_layers)}')
    print(f'Freezing layers: {len(model_layers) - len(parameters)}')
    print(f'Fine-tuning layers: {len(parameters)}')
    print(f'Requested to fine-tune: {num_to_freeze}')
    
    return model
# def apply_custom_transfer_learning(model, num_to_freeze):
#     model_layers = [name for name,para in model.named_parameters()]

#     for ii in range(42):
#             if ii == 0:
#                 layers = [','.join(model_layers[:3])]
#             elif ii <= 3 and ii > 0:
#                 layers += [','.join(model_layers[(ii*4):(ii*4+4)])]
#             elif ii <= 39 and ii > 3:
#                 layers += [','.join(model_layers[(16+(ii-4)*9):(25+(ii-4)*9)])]
#             else:
#                 layers += [','.join(model_layers[(340+(ii-40)*2):(342+(ii-40)*2)])]
#     fine_tune_layers = ','.join(layers[num_to_freeze-len(layers):]).split(',')
#     for name, param in model.named_parameters():
#         #print(name)
#         if name not in fine_tune_layers:
#             #print(name)
#             param.requires_grad = False

#     parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
#     print(f'The number of parameters are {len(parameters)}')
#     print(f'The number of layers to fine tune are {len(fine_tune_layers)}')
#     # print([len(parameters), len(fine_tune_layers)])
#     assert len(parameters) == len(fine_tune_layers)

#     return model