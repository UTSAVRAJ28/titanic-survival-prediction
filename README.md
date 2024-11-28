# Titanic Survival Prediction

## Overview
This project aims to predict the survival of Titanic passengers using machine learning models. The models used include Support Vector Machine (SVM), Neural Network (MLPClassifier), and Random Forest. The project involves data preprocessing, hyperparameter tuning, and model evaluation, with Random Forest emerging as the most accurate model with an 81% accuracy rate.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Models Used](#models-used)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Titanic-Survival-Prediction.git
    cd Titanic-Survival-Prediction
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2. Open the `project.ipynb` file and run the cells to see the full analysis.

## Project Structure

- `project.ipynb`: The main notebook containing all the code, analysis, and results.
- `data/`: Directory to store the dataset.
- `models/`: Directory where trained models and results are stored.
- `requirements.txt`: File listing the Python dependencies required for the project.
- `README.md`: Project documentation and instructions.

## Data Preprocessing

The project involves several data preprocessing steps:
- Imputation of missing values in 'Age' and 'Fare' columns.
- Categorical encoding of 'Sex' and 'Embarked' columns.
- Column pruning to remove irrelevant features.
- Train-test split with 80% training and 20% validation data.

## Models Used

1. **Support Vector Machine (SVM):**
   - Tuned using GridSearchCV.
   - Optimal hyperparameters: `C=10`, `gamma=0.01`, `kernel='linear'`.

2. **Neural Network (MLPClassifier):**
   - Configured with hidden layers and activation functions.
   - Tuned using GridSearchCV.
   - Optimal hyperparameters: `hidden_layer_sizes=(100, 100)`, `activation='tanh'`.

3. **Random Forest:**
   - Optimal hyperparameters: `n_estimators=100`, `max_depth=None`, `min_samples_split=2`.
   - Achieved the highest accuracy of 81%.

## Results

- **Model Performance:**
  - SVM: 80% accuracy.
  - Neural Network: 78% accuracy.
  - Random Forest: 81% accuracy.

- **Feature Importance:** Random Forest was used to assess feature importance, providing insights into which features were most influential in predicting survival.

## Future Work

Potential improvements include:
- Enhanced feature engineering.
- Exploration of ensemble methods.
- Experimentation with advanced algorithms.
- Application of data scaling techniques.
- Cross-validation strategies to improve model robustness.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.


