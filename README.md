# CNN Image Recognition

This project implements a Convolutional Neural Network (CNN) for rice image classification using PyTorch, torchvision and matplotlib to plot the results.

## Dataset

- The dataset should be placed in the `Rice_Image_Dataset/` directory, with subfolders for each rice class (e.g., Arborio, Basmati, Ipsala, Jasmine, Karacadag).
- Images are automatically resized to 32x32 pixels and normalized.
- The Rice Image Dataset, created by Murat Koklu, is publicly available in the public domain and was sourced from Kaggle.

## Model Architecture

- The model is defined in [`NeuralNet`](main.py), consisting of:
  - 3 convolutional layers with batch normalization and max pooling
  - 3 fully connected layers
  - Output layer with 5 classes

## Training

- The dataset is split into training (70%), validation (15%), and test (15%) sets.
- Data augmentation is applied to the training set.
- The model is trained for 10 epochs using SGD optimizer and cross-entropy loss.
- Training and validation loss are plotted and saved as `loss_plot.png`.

## Evaluation

- After training, the model is evaluated on the test set.
- Test accuracy is printed at the end.

## Usage

1. Place your rice image dataset in the `Rice_Image_Dataset/` folder.
2. Run the script:

   ```sh
   python main.py
   ```

3. Check the output for training progress, validation accuracy, and final test accuracy.
4. The trained model weights are saved to `trained_net.pth`.

## Requirements

- Python 3.13.6
- PyTorch 2.80+cu126
- torchvision
- matplotlib

Install dependencies with:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
pip install matplotlib
```

It is highly recommended to use CUDA for this project due to the large volume of data—over 15,000 images per class with a batch size of 32. This will ensure more efficient and rapid processing.

## Files

- [`main.py`](main.py): Main training and evaluation script
- `Rice_Image_Dataset/`: Dataset folder
- `trained_net.pth`: Saved model weights
- `loss_plot.png`: Training/validation loss plot

## Results

The `loss_plot.png` demonstrates a highly effective training process. Both training and validation losses exhibit a sharp initial decrease before converging at low, stable values. This pattern indicates that the model learns efficiently and generalizes well. 

Furthermore, the low initial loss suggests the dataset features are clearly separable, making the classification task highly tractable for the chosen architecture.

<div align="center">
  <img align="center" src= "https://github.com/Italohek/CNN-Image-Recognition/blob/main/loss_plot.png" width="720" />
</div>
