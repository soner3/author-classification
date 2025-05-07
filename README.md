# Author Identification using Text Mining and Machine Learning

This project aims to classify texts based on their authorship using traditional ML models and deep learning approaches (BERT). The dataset consists of multiple authors and their respective documents. The Python version used for this project is 3.12.9

## Folder Structure

- `data/raw/dataset_authorship/`: Original dataset
- `data/processed/`: Cleaned and labeled data
- `src/`: Source code for data processing, feature extraction, and modeling
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation

## Getting Started

Follow the steps below to set up and run the author identification project.

### 1. Prepare the dataset

Open the ZIP dataset_authorship.zip and put the folder dataset_authorship in the folder ./data/raw/. The folder must have the name `dataset_authorship` and it must be in the raw folder of the data directory otherwise the project will not find the dataset

### 2. Create and Activate the Virtual Environment

Ensure that you are in the **project root directory**, then create activate the virtual environment with the following commands:

**On Windows:**

```bash
python -m venv venv
```

```bash
./venv/Scripts/activate
```

```bash
pip install -r requirements.txt
```

> This will create and enable the Python environment with all required dependencies.

### 3. Run Tests

With the environment still activated, execute the following command to run all unit tests:

```bash
pytest
```

> 🧪 This ensures that all modules are working as expected before proceeding.

### 4. Execute the Notebooks

Use the same activated virtual environment to run the Jupyter notebooks in the `notebooks/` folder.

- Open Jupyter Lab or Notebook:

  ```bash
  jupyter lab
  ```

  or

  ```bash
  jupyter notebook
  ```

- Then **execute the notebooks in order**:
  1. `01_*.ipynb`
  2. `02_*.ipynb`
  3. `03_*.ipynb`

Each notebook builds on the previous one, and together they perform the complete pipeline: preprocessing, modeling, evaluation and visualization.
