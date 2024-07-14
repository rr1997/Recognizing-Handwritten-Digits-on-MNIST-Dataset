# Recognizing-Handwritten-Digits-on-MNIST-Dataset
Using k-Nearest Neighbor

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## K-Nearest Neighbor Classifier from Scratch

Implementation of K-Nearest Neighbors classifier from scratch for image classification on the MNIST dataset. No existing sklearn packages were used for writing the knn code.

## Dataset

In the MNIST dataset, each sample is a picture of a handwritten digit. Each sample includes 28x28 grey-scale pixel values as features and a categorical class label out of 0-9. For more details, please refer to Dr. Yann Lecun’s page (http://yann.lecun.com/exdb/mnist/). The dataset has been imported using the sklearn.dataset package.

![MNIST Dataset]([https://upload.wikimedia.org/wikipedia/commons/2/2a/MnistExamples.png](https://www.google.com/search?q=mnist+dataset+images&sca_esv=954b519294f2b658&sca_upv=1&rlz=1C1CHBF_enIN1019IN1019&sxsrf=ADLYWIIxF3d8yPEhYwPitC90BBqbbQGx8g%3A1720978719092&ei=Hw2UZoKuBYm74-EPmsyAiAI&oq=mnist+dataset+ima&gs_lp=Egxnd3Mtd2l6LXNlcnAiEW1uaXN0IGRhdGFzZXQgaW1hKgIIATIKECMYgAQYJxiKBTILEAAYgAQYkQIYigUyCxAAGIAEGJECGIoFMgsQABiABBiRAhiKBTIGEAAYFhgeMgYQABgWGB4yCBAAGBYYHhgPMgsQABiABBiGAxiKBTILEAAYgAQYhgMYigUyCxAAGIAEGIYDGIoFSNoYUKQBWKoPcAF4AZABAJgBeaABuQOqAQMwLjS4AQPIAQD4AQGYAgWgAtoDwgIHECMYsAMYJ8ICChAAGLADGNYEGEfCAg0QABiABBiwAxhDGIoFwgIFEAAYgATCAgoQABiABBgUGIcCmAMAiAYBkAYKkgcDMS40oAedHw&sclient=gws-wiz-serp#imgrc=Xe5QCtIozPkSNM&imgdii=3F5_2NNe6lmbgM))

## Implementation

I have implemented the classifier with the L2-norm (Euclidean distance) as the distance measurement between samples. In the original dataset, the first 60,000 samples are for training, and the remaining 10,000 samples are for testing. In this implementation, I have used the first 6,000 samples from the original training set for training KNN, and the first 1,000 from the original test set for testing KNN.

## Project Structure

The project is organized as follows:

Recognizing-Handwritten-Digits-on-MNIST-Dataset/
├── data/
│ ├── mnist_train.csv
│ ├── mnist_test.csv
├── models/
│ ├── knn_model.pkl
├── notebooks/
│ ├── Data_Preprocessing.ipynb
│ ├── Model_Training.ipynb
│ ├── Model_Evaluation.ipynb
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
├── README.md
├── requirements.txt


- `data/`: Contains the dataset files.
- `models/`: Stores the trained model.
- `notebooks/`: Contains Jupyter notebooks for data preprocessing, model training, and evaluation.
- `src/`: Contains the source code for data preprocessing, model training, and evaluation.

## Installation

To run this project, you need to have Python installed. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt

Usage
Data Preprocessing:

Navigate to the notebooks/ directory and open Data_Preprocessing.ipynb.
Run the notebook to preprocess the data.
Model Training:

Open Model_Training.ipynb in the notebooks/ directory.
Run the notebook to train the KNN model.
Model Evaluation:

Open Model_Evaluation.ipynb in the notebooks/ directory.
Run the notebook to evaluate the trained model.
Alternatively, you can run the Python scripts in the src/ directory:

python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py

Model
The model used in this project is K-Nearest Neighbors (KNN). KNN is a simple, instance-based learning algorithm that assigns a class to a new sample based on the majority class among its k-nearest neighbors.

Results
The model achieves an accuracy of approximately 97% on the test set. Below are some example predictions:



Contributing
Contributions are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.
