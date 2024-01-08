# News Classification using CatBoost

This repository contains scripts for training and inference in a machine
learning project focused on classifying news texts. It leverages the powerful
CatBoost library to perform classification, utilizes DVC for data versioning and
MLFlow for logging.

This is an educational project, the focus of which is on studying MLOps
approaches when developing a service based on machine learning. Therefore, the
part responsible for collecting data and training the model is done quite
formally. However, in the next stages I plan to also finalize the model and
scripts responsible for collecting news data.

As a data source for training the model, I used the well-known dataset
[The 20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), which contains about
20 000 news articles, divided into 20 different newsgroups. This data can be
easily obtained in prepared form using
[sklearn](https://scikit-learn.org/stable/) library. But, for educational
purposes, I saved the prepared data in `json` files in order, on the one hand,
to bring the situation a little closer to the real one, but on the other hand,
to save time on independently collecting and preparing data.

The trained model should predict which of 20 categories the text from the test
sample most likely belongs to.

## Project Structure

The project is structured with two main Python scripts at its root:

- `train.py`: This script handles the downloading of the training dataset,
  trains the CatBoost model, and then saves the trained model parameters to the
  disk. Training parameters and resulting metrics are logged using MLFlow
- `infer.py`: This script takes care of loading the trained model from the disk,
  downloading the test dataset, performing predictions, and finally outputting
  it to the csv file.

## How to Use

To run the scripts, you will need to have Python installed along with the
required packages. You can set up your environment by following these steps:

1. Clone the repository: `git clone <repository-url>`
2. Navigate into the project directory: `cd news-classification`
3. Install dependencies in virtual environment: `poetry install`
4. Activate the virtual environment: `poetry shell`
5. Install pre-commit: `pre-commit install`
6. If necessary, make changes to the settings files located in the directory
   `configs` (see description of settings files in the next section)
7. Start the MLFlow server: `mlflow ui`
8. To train the model, simply execute the train.py script with Python. This will
   automatically handle downloading the dataset, training the model, and saving
   it: `python train.py`
9. Upon successful completion the files `catboost_model.cbm` and
   `vectorizer_data.json` will appear in the `models` directory
10. To explore the metrics, open MLFlow web interface at
    [http://127.0.0.1:5000](http://127.0.0.1:5000) by default
11. To make predictions with the trained model and assess its performance,
    execute the infer.py script: `python infer.py`
12. This script loads the trained model, fetches the test dataset, performs
    predictions, and then save it to the `prediction.csv` file in the
    `prediction` directory

Please make sure to keep the data directory synchronized with the DVC remote
storage to ensure consistency and reproducibility across different machines and
setups.

## Project config

All project config files are located in directory `configs` and contain the
following files with parameters listed below:

- `main_conf.yaml`
  - `urls`
    - `mlflow_url`: <url>: URL where the MLFlow server is accessible
  - `params`: model training parameters
    - `iterations`
    - `learning_rate`
    - `tree_depth`
- `files/dataset_files.yaml`: file names with training and test datasets and
  model settings
  - `test_data`
  - `test_targets`
  - `train_data`
  - `train_targets`
  - `vectorizer`
  - `model`

## License

This project is open-sourced under the MIT License. See the LICENSE file for
more information.
