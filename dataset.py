from imports import *

class BreastDataset(Dataset):
    def __init__(self, df, transforms=None, augment_transforms=None):
        self.paths = df['path'].values
        self.labels = df['cancer'].values
        self.datasettype = df['datasettype'].values
        self.transforms = transforms
        self.augment_transforms = augment_transforms  # Augmentations for label==1 only

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load image
        image_path = self.paths[index]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image is None (failed to load).")
            
            # Convert BGR (default by cv2) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Retrieve label and dataset type
            label = self.labels[index]
            datasettype = self.datasettype[index]

            # Apply augmentation transforms for cancer-positive images
            if label == 1 and self.augment_transforms:
                image = self.augment_transforms(image=image)["image"]
            elif self.transforms:
                image = self.transforms(image=image)["image"]

        except Exception as e:
            # Raise the exception to stop execution if needed
            raise ValueError(f"Error loading or processing image: Error: {e}")

        return image, label, datasettype, index
