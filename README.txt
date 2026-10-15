MNIST ML examples — each example is a separate Python script

01_explore_mnist.py       View digit samples and class balance.
02_knn_classifier.py      k-nearest-neighbors, confusion matrix, mistakes.
03_logistic_regression.py Multiclass linear model and pixel-weight maps.
04_mlp_pytorch.py         Fully connected neural network and learning curves.
05_cnn_pytorch.py         CNN, learning curves, confusion matrix, filters.
06_pca_representation.py PCA dimensionality reduction and 2D scatter plot.

Install dependencies:
    pip install torch torchvision matplotlib numpy scikit-learn

Run from this directory, for example:
    python 01_explore_mnist.py

The first run downloads MNIST into ./data. Figures are shown and saved as PNG
files in the current working directory. PyTorch examples use CUDA if available,
otherwise CPU. Neural-network examples use subsets and five epochs for a quick
run; increase subset sizes or epochs for stronger results.
