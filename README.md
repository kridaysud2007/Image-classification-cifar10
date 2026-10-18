# CIFAR-10 Image Classification

This project trains a convolutional neural network (CNN) to classify images
from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
The main project workflow is in [`model.ipynb`](./model.ipynb): it downloads
and prepares the data, trains the CNN, tracks validation performance, and
evaluates the saved model on the test set. A small Streamlit application is
also included for trying the trained model on an uploaded image or an image
URL.

## What the notebook does

### 1. Imports and dataset preparation

The notebook uses PyTorch and Torchvision for the model, dataset and image
transforms, Matplotlib for inspecting images, and NumPy and Seaborn to
calculate and plot the confusion matrix. It loads CIFAR-10 with
`torchvision.datasets.CIFAR10`. The dataset has 10 object categories and
32 × 32 RGB images.

Torchvision downloads the data into `data/` the first time the notebook runs.
That directory is local working data and is intentionally excluded from Git.

### 2. Image transforms

The notebook defines separate transformations for training and evaluation:

- Both convert images to floating-point tensors in the range `[0, 1]` and
  normalize each RGB channel using the CIFAR-10 channel means
  `[0.4914, 0.4822, 0.4465]` and standard deviations
  `[0.2023, 0.1994, 0.2010]`.
- Training images also receive a random padded crop and random horizontal
  flip. These augmentations expose the model to varied versions of training
  images and can improve generalization.
- Validation and test images do not receive random augmentation, so their
  evaluation is repeatable.

The training and evaluation datasets use the same underlying CIFAR-10
training images but have different transforms. A seeded random permutation
creates a reproducible split: 45,000 images for training and 5,000 for
validation. The official 10,000-image test set remains separate.

### 3. Data loaders and device

The training loader uses batches of 64 and shuffles the training examples.
Validation and test loaders also use batches of 64 but do not shuffle.

The notebook selects the available PyTorch accelerator (the recorded run used
Apple MPS) and moves each batch and the model to that device. PyTorch can also
run the model on CPU or CUDA-capable hardware when available.

### 4. CNN architecture

The `NeuralNetwork` class contains four 3 × 3 convolution layers arranged in
two blocks:

1. A first block maps the 3 input color channels to 32 feature channels,
   applies batch normalization and ReLU, then repeats the 32-channel
   convolution before max pooling.
2. A second block increases the feature channels from 32 to 64, applies
   batch normalization and ReLU, repeats the 64-channel convolution, then
   max-pools again.
3. The two pooling operations reduce the spatial size from 32 × 32 to 8 × 8.
   The resulting `64 × 8 × 8` features are flattened.
4. A fully connected layer maps those features to 128 units, followed by
   ReLU and dropout (`p=0.3`). The final layer outputs 10 scores, one for each
   CIFAR-10 class.

The network returns raw class scores (logits). `CrossEntropyLoss` consumes
these directly during training; a softmax is only needed when displaying
probabilities in the app.

### 5. Training and validation

The notebook trains with `CrossEntropyLoss` and the Adam optimizer at a
learning rate of `0.001` for 100 epochs. The training loop puts the model in
training mode, computes the loss for each batch, backpropagates the gradients,
and updates the weights. It prints a training-loss sample every 100 batches.

After each epoch, the evaluation function switches the model to evaluation
mode and measures loss and classification accuracy on the validation set
without calculating gradients. Validation results are used to understand
training progress and compare model changes.

After training, the notebook evaluates on the separate test set and saves the
model parameters to `cifar10_cnn.pth`. The included checkpoint corresponds
to the model architecture in `model_def.py`. The notebook also plots an
un-normalized confusion matrix for the test set, showing the number of
predictions in each true/predicted class pair and which classes the model
most often confuses.

### Recorded results

The saved notebook output records **84.86% validation accuracy at epoch 100**
and **83.84% test accuracy** for one training run. These are reference results
from that run, not guarantees: training can vary with initialization, data
order and software/hardware versions.

Use validation performance when tuning the architecture and training
settings. Keep the test set for a final evaluation after those choices are
made.

## Run the notebook

Use Python 3.10 or newer. From the project directory, create and activate a
virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Open `model.ipynb` in Jupyter or VS Code and run the cells from top to bottom.
The dataset will be downloaded automatically if it is not already present.
Running the final cells retrains the model and writes `cifar10_cnn.pth` in the
project directory.

## Try the interactive app

The optional Streamlit app loads `cifar10_cnn.pth` and provides top-three
predictions for an uploaded image or an image URL. Start it from the project
directory after installing the requirements:

```bash
streamlit run app.py
```

The app resizes input images to 32 × 32 and applies the same channel
normalization used during evaluation. CIFAR-10 consists of small images, so
predictions on ordinary high-resolution photographs may be less reliable than
on images similar to the dataset. When the app is deployed publicly, image
URLs should be restricted to trusted sources.

## Project files

| File | Purpose |
| --- | --- |
| `model.ipynb` | Dataset preparation, visualization, CNN training, validation and test evaluation |
| `model_def.py` | CNN architecture shared with the app |
| `cifar10_cnn.pth` | Trained model parameters used by the app |
| `app.py` | Optional Streamlit interface for image classification |
| `requirements.txt` | Python package dependencies |

The downloaded dataset, virtual environment, Python cache files and Jupyter
checkpoint directory are excluded by `.gitignore`.
