# Customer Churn Prediction

A machine learning project that predicts customer churn for a bank using Artificial Neural Networks (ANN).

## Project Overview

This project uses a deep learning model to predict whether a bank customer is likely to leave the bank (churn) based on various customer attributes. The model is trained on historical customer data and can be used to identify customers who are at risk of churning, allowing the bank to take proactive retention measures.

## Features

- Data preprocessing and exploration in Jupyter notebooks
- Neural network model built with TensorFlow/Keras
- Interactive web application built with Streamlit
- Model evaluation and performance metrics
- Saved model and preprocessing components for easy deployment

## Dataset

The project uses the "Churn_Modelling.csv" dataset which contains the following features:
- Customer ID and demographic information (Geography, Gender, Age)
- Banking relationship details (Tenure, Balance, Number of Products)
- Customer behavior (Credit Score, Has Credit Card, Is Active Member)
- Target variable: Exited (whether the customer left the bank)

## Project Structure

```
├── app.py                         # Streamlit web application
├── experiments.ipynb              # Jupyter notebook with model development
├── prediction.ipynb               # Jupyter notebook for making predictions
├── model.h5                       # Saved neural network model
├── scaler.pkl                     # StandardScaler for numerical features
├── onehot_encoder_geo.pkl         # OneHotEncoder for Geography feature
├── label_encoder_gender.pkl       # LabelEncoder for Gender feature
├── Churn_Modelling.csv            # Original dataset
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

To run the Streamlit web application:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser.

### Exploring the Notebooks

The project includes two Jupyter notebooks:
- `experiments.ipynb`: Contains the data exploration, preprocessing, model development, and evaluation
- `prediction.ipynb`: Demonstrates how to use the trained model for making predictions

To open the notebooks:

```bash
jupyter notebook
```

## Model Performance

The neural network model achieves:
- Accuracy: ~86%
- Precision: ~80%
- Recall: ~75%
- F1 Score: ~77%

## Future Improvements

- Implement hyperparameter tuning to optimize model performance
- Add more advanced feature engineering
- Explore other model architectures (LSTM, GRU)
- Deploy the application to a cloud platform

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Kaggle](https://www.kaggle.com/datasets)
- Inspired by various churn prediction research papers and projects 