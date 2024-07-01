# Linear Regression Model for Car Price Prediction

Welcome to the Linear Regression Model project! This project focuses on building and analyzing a linear regression model to predict car prices based on various features such as mileage and year. Below, you'll find an overview of the project, including setup instructions, usage, and a brief explanation of the code.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Model Implementation](#model-implementation)
  - [Data Analysis](#data-analysis)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project utilizes a custom linear regression model implemented in Python to predict the prices of cars. The data is preprocessed and analyzed using various Python libraries, including NumPy, Pandas, Matplotlib, and Seaborn. The model is trained on a dataset of car listings, and the performance is evaluated using various metrics.

## Setup Instructions

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/recabet/LinearRegressionPY.git
    ```
2. Navigate to the project directory:
    ```bash
    cd linear-regression-car-price
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Make sure you have the dataset (`c_class_listings.csv`) in the project directory.

## Usage

1. Run the `dataset-analysis.py` script to preprocess the data and generate visualizations:
    ```bash
    python dataset-analysis.py
    ```
2. The model is implemented in `model.py` and can be used to train and predict car prices. You can import the `LinearRegression` class from `model.py` and use it in your scripts.

## Code Explanation

### Model Implementation

The linear regression model is implemented in `model.py`. Below is an overview of the class and its methods:

## Code Explanation

### Model Implementation

The linear regression model is implemented in `model.py`. Below is an overview of the class and its methods:

- **`__init__(self, rate=0.01, iter=1000)`**:
  This is the constructor for the `LinearRegression` class. It initializes the learning rate (`rate`), the number of iterations for training (`iter`), weights (`w`), bias (`b`), and statistical measures for normalization (`X_mean`, `X_std`, `y_mean`, `y_std`).

- **`__normalize(self, X, y)`**:
  This private method normalizes the input features (`X`) and target values (`y`). It calculates the mean and standard deviation of `X` and `y`, then uses these statistics to standardize the data. This is crucial for ensuring that the model converges properly during training.

- **`fit(self, X, y)`**:
  This method trains the linear regression model on the provided dataset (`X`, `y`). It first normalizes the data, then iteratively updates the weights and bias using gradient descent. The method returns the trained weights and bias.

- **`compute_cost(self, X, y)`**:
  This method calculates the cost (or loss) of the model on the given dataset. It predicts the output for `X` and computes the mean squared error between the predicted and actual values. This helps in evaluating how well the model is performing.

- **`predict(self, X)`**:
  This method predicts the target values for the given input features (`X`) using the trained weights and bias. It's used to make predictions on new data after the model is trained.

- **`r_squared(self, X, y)`**:
  This method computes the R-squared value, a statistical measure that represents the proportion of the variance for the target variable that is explained by the input features. It gives an indication of the goodness of fit of the model.

- **`error(self, X, y)`**:
  This method calculates the percentage error between the predicted and actual values. It provides an additional way to evaluate the model's performance.

### Data Analysis

The data analysis is conducted in `dataset-analysis.py`. Below is an overview of the script:

- **Data Preprocessing**:
  The script reads the dataset `c_class_listings.csv`, cleans it, and preprocesses it for analysis. This includes converting prices to a uniform currency, removing irrelevant columns, and handling categorical variables.

- **Visualizations**:
  Several plots are generated to visualize the relationships between different features and the target variable (price). These include scatter plots of mileage vs. price and year vs. price, and a histogram of mileage with a normal distribution curve overlay.

- **Training the Model**:
  The script trains the linear regression model on the preprocessed data. It then visualizes the model's predictions in 3D space to illustrate the fit of the regression plane to the data points.

## Results

The results section can include a summary of the model's performance, including metrics such as R-squared and error percentage. You can also include visualizations generated during data analysis and model training.

## Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes. Feel free to open issues for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
