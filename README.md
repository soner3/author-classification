# Author Identification using Text Mining and Machine Learning

This project aims to classify texts based on their authorship using traditional ML models and deep learning approaches (BERT). The dataset consists of multiple authors and their respective documents.

## Folder Structure

- `data/raw/dataset_authorship/`: Original dataset
- `data/processed/`: Cleaned and labeled data
- `src/`: Source code for data processing, feature extraction, and modeling
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation

Natürlich! Hier ist die komplette README mit dem ergänzten **„Getting Started“-Teil**, fertig formatiert in Markdown – einfach kopieren und in deine `README.md` einfügen:

````markdown
# Author Identification using Text Mining and Machine Learning

This project aims to classify texts based on their authorship using traditional ML models and deep learning approaches (BERT). The dataset consists of multiple authors and their respective documents.

## Folder Structure

- `data/raw/dataset_authorship/`: Original dataset
- `data/processed/`: Cleaned and labeled data
- `src/`: Source code for data processing, feature extraction, and modeling
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation

## Getting Started

Follow the steps below to set up and run the author identification project.

### 1. Activate the Virtual Environment

Ensure that you are in the **project root directory**, then activate the virtual environment:

**On Windows:**

```bash
./venv/Scripts/activate
```
````

> This will enable the Python environment with all required dependencies.

### 2. Run Tests

With the environment still activated, execute the following command to run all unit tests:

```bash
pytest
```

> 🧪 This ensures that all modules are working as expected before proceeding.

### 3. Execute the Notebooks

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
