"""
Configuration constants for the mammography AI project.
"""

# Paths
INPUT_CHK_PT = "/Users/mukulsherekar/pythonProject/MammographyAI/CHECKPOINTS/singleviewExOri.pth"

# Model parameters
THRESHOLD = 0.3

# Image normalization
NORMALIZE_MEAN = [77.52425988, 77.52425988, 77.52425988]
NORMALIZE_STD = [51.8555656, 51.8555656, 51.8555656]

# Training parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_EPOCHS = 5
