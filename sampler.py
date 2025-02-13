from imports import *

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        """
        Custom sampler to ensure each batch has an equal number of positive and negative samples.
        
        Args:
            labels (list or array): List of class labels for each sample in the dataset (0 for negative, 1 for positive).
            batch_size (int): The desired batch size. Must be even for equal sampling.
        """
        assert batch_size % 2 == 0, "Batch size must be even to balance positive and negative samples."
        
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]
        self.num_batches = min(len(self.positive_indices), len(self.negative_indices)) * 2 // batch_size
    
    def __len__(self):
        """
        Return the number of batches.
        """
        return self.num_batches
    
    def __iter__(self):
        """
        Generate balanced batches.
        """
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        # Create batches
        pos_batches = np.array_split(self.positive_indices, self.num_batches)
        neg_batches = np.array_split(self.negative_indices, self.num_batches)
        
        balanced_batches = []
        for pos, neg in zip(pos_batches, neg_batches):
            batch = np.concatenate((pos, neg))
            np.random.shuffle(batch)
            balanced_batches.append(batch)
        
        return iter(balanced_batches)
    
