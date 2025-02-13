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



def load_data_from_csv(args, csv_pth, training_flag=False):
    df = pd.read_csv(csv_pth)

    if training_flag and args.dataset_partition:
        df = pd.read_csv(csv_pth, nrows=args.rows_experiment)
        print(f'Loading {args.rows_experiment} rows for experimentation')
        print(df.shape)
    else:
        print(f'Loading the whole dataset for a full final run')
        df = pd.read_csv(csv_pth)
        print(df.shape)
    
    if training_flag and (args.finite_sample_rate is not None):
        df = data_sampling(df, args.finite_sample_rate)
        df.to_csv(os.path.join(args.out_dir, f'training_list_sample_rate_{args.finite_sample_rate}.csv'), index=False)
    
    if args.data == 'EMBED':
        df['path'] = df["path_eq"]
    else:
        raise RuntimeError('!ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
    
    # Create dataset with or without augmentation
    if training_flag:
        if args.training_augment:
            dataset = BreastDataset(df,transforms=get_transforms(augment=False),augment_transforms=get_transforms(augment=True))
            print("Training with selective augmentation for cancer-positive images")
        else:
            dataset = BreastDataset(df, transforms=get_transforms(augment=False))
    else:
        dataset = BreastDataset(df, transforms=get_transforms(augment=False))

    return dataset