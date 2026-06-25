========================================================
  Deep Learning - Spring 2025 | Assignment 5 (Bonus)
  Image Generation Using Diffusion Models
  Student: Neeka Javed | Roll No: MSDS25008
========================================================

----------------------------------------------------------------
DIRECTORY STRUCTURE
----------------------------------------------------------------

Neeka_MSDS25008_05/
|
|-- MSDS25008_05.py               # Main training script
|-- MSDS25008_05_allCode.py       # All code combined in one file
|-- test_single_sample.ipynb      # Notebook to load model and generate images
|-- Report.pdf                    # Detailed report with results
|-- Readme.txt                    # This file
|
|-- Models/
|   |-- best_model.pth            # Best model (saved at epoch 88)
|   |-- model_epoch_10.pth
|   |-- model_epoch_20.pth
|   |-- ...
|   |-- model_epoch_100.pth
|
|-- Results/
|   |-- generated_sample.png      # Final generated images
|   |-- generated_grid.png        # Generated images in grid format
|   |-- noise_propagation.png     # Forward noise progression visualization
|   |-- training_loss_graph.png   # Loss curve over epochs
|   |-- generated_epoch_10.png
|   |-- generated_epoch_20.png
|   |-- ...
|   |-- generated_epoch_100.png

----------------------------------------------------------------
SOURCE FILES
----------------------------------------------------------------

dataset.py      -- AnimalDataset class (DataLoader)
Diffusion.py    -- Forward process, reverse sampling
U_NET.py        -- UNet denoising model architecture

----------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------

Python       >= 3.8
PyTorch      >= 1.12
torchvision  >= 0.13
Pillow       >= 9.0
matplotlib   >= 3.5
numpy        >= 1.21
jupyter      >= 1.0  (for test_single_sample.ipynb)

Install all dependencies:
    pip install torch torchvision pillow matplotlib numpy jupyter

----------------------------------------------------------------
DATASET SETUP
----------------------------------------------------------------

1. Place the animal dataset in a folder named: animal_data/
2. Each animal class should be in its own subfolder:

   animal_data/
   |-- bear/
   |   |-- image1.jpg
   |   |-- image2.jpg
   |   |-- ...
   |-- cat/
   |-- dog/
   |-- elephant/
   |-- horse/

3. The code uses 5 classes with a maximum of 20 images per class.
4. Supported formats: .jpg, .jpeg, .png

----------------------------------------------------------------
HOW TO TRAIN
----------------------------------------------------------------

Basic usage (default dataset path = "animal_data"):
    python MSDS25008_05.py

Custom dataset path:
    python MSDS25008_05.py --dataset_path /path/to/your/animal_data

Additional command line arguments:
    --dataset_path     Path to dataset folder         (default: animal_data)
    --epochs           Number of training epochs      (default: 200)
    --batch_size       Batch size                     (default: 16)
    --lr               Learning rate                  (default: 0.0001)
    --T                Number of diffusion timesteps  (default: 1000)
    --patience         Early stopping patience        (default: 20)

Example with all arguments:
    python MSDS25008_05.py --dataset_path animal_data --epochs 200 --batch_size 16 --lr 0.0001

Training outputs:
    - Models saved to:  Models/
    - Images saved to:  Results/
    - Best model:       Models/best_model.pth
    - Loss graph:       Results/training_loss_graph.png

----------------------------------------------------------------
HOW TO GENERATE IMAGES (After Training)
----------------------------------------------------------------

Option 1 - Python script:
    python MSDS25008_05.py --test_only

Option 2 - Jupyter Notebook (for viva evaluation):
    jupyter notebook test_single_sample.ipynb
    Run all cells to load the best model and generate images.

Generated images are saved to: Results/generated_sample.png

----------------------------------------------------------------
HOW TO VISUALIZE FORWARD NOISE PROGRESSION
----------------------------------------------------------------

    python visualize_noise.py

This will generate: Results/noise_propagation.png
Showing the image at timesteps: t=0, 50, 100, 200, 400, 600, 800, 999

----------------------------------------------------------------
TRAINING DETAILS
----------------------------------------------------------------

- Architecture    : U-Net with skip connections
- Normalization   : GroupNorm (32 groups)
- Activation      :     ReLU 
- Loss Function   : Custom MSE Loss
- Optimizer       : Adam (lr=1e-4)
- Noise Schedule  : Linear beta schedule (1e-4 to 0.02)
- Diffusion Steps : T = 1000
- Image Size      : 64 x 64
- Training ran for 108 epochs (early stopping)
- Best model saved at epoch 88 (loss = 0.0316)

----------------------------------------------------------------
GITHUB
----------------------------------------------------------------

Repository contains regular commits reflecting:
- Dataset loading implementation
- Forward diffusion process
- U-Net architecture
- Training loop
- Testing and evaluation
- Bug fixes and results

