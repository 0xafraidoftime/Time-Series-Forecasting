# MLP for Univariate Time Series Forecasting

This repository contains implementations of different neural network architectures for univariate time series forecasting, including Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN) models.

## Overview

The project demonstrates how to:
- Transform time series data into supervised learning problems
- Implement and compare multiple neural network architectures for forecasting
- Evaluate model performance on datasets with trend and seasonality patterns

## Features

- **Multi-Layer Perceptron (MLP)**: Simple feedforward neural network for time series prediction
- **Convolutional Neural Network (CNN)**: Uses 1D convolutions to capture local patterns in time series
- **Recurrent Neural Network (RNN)**: Captures sequential dependencies in time series data
- **Performance Comparison**: Visual comparison of training losses across all models
- **Flexible Data Generation**: Synthetic data generation with trend and seasonality components

## Requirements

```
numpy
keras
tensorflow
matplotlib
scikit-learn
```

## Installation

1. Clone this repository:
```bash
git clone <https://github.com/0xafraidoftime/Time-Series-Forecasting>
cd mlp-time-series-forecasting
```

2. Install required packages:
```bash
pip install numpy keras tensorflow matplotlib scikit-learn
```

## Usage

### Basic MLP Example

The code includes a simple example with a basic sequence:

```python
# Define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Choose number of time steps for lookback window
n_steps = 3

# Transform to supervised learning problem
X, y = split_sequence(raw_seq, n_steps)

# Train MLP model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1000, verbose=0)
```

### Advanced Model Comparison

The project also includes a comprehensive comparison of three different architectures on synthetic data with trend and seasonality:

```python
# Generate synthetic data with trend and seasonality
raw_seq = generate_seasonal_trend_data(100)

# Train and compare MLP, CNN, and RNN models
# Results are visualized in a training loss comparison plot
```

## Model Architectures

### MLP (Multi-Layer Perceptron)
- **Input Layer**: Dense layer with 100 neurons and ReLU activation
- **Output Layer**: Single neuron for regression
- **Best for**: Simple patterns and baseline comparisons

### CNN (Convolutional Neural Network)
- **Conv1D Layer**: 64 filters with kernel size 2
- **MaxPooling**: Pool size 2 for dimensionality reduction
- **Dense Layers**: 50 neurons + output layer
- **Best for**: Capturing local patterns and short-term dependencies

### RNN (Recurrent Neural Network)
- **SimpleRNN Layer**: 50 units with ReLU activation
- **Output Layer**: Single neuron for regression
- **Best for**: Capturing sequential dependencies and temporal patterns

## Key Functions

### `split_sequence(sequence, n_steps)`
Transforms a univariate time series into a supervised learning dataset.

**Parameters:**
- `sequence`: Input time series data
- `n_steps`: Number of previous time steps to use as input features

**Returns:**
- `X`: Input sequences of shape [samples, n_steps]
- `y`: Target values

### `generate_seasonal_trend_data(n_steps)`
Generates synthetic time series data with both trend and seasonal components.

**Parameters:**
- `n_steps`: Length of the time series to generate

**Returns:**
- Numpy array with synthetic time series data

## Results

The models are trained on the same dataset and their performance can be compared through:
1. **Training Loss Curves**: Visual comparison of how each model learns over epochs
2. **Prediction Accuracy**: Direct comparison of predictions on test inputs
3. **Model Summaries**: Architecture details and parameter counts

## File Structure

```
├── mlp_for_univariate_time_series_forecasting.py  # Main Python script
├── mlp_for_univariate_time_series_forecasting.ipynb  # Jupyter notebook version
└── README.md  # This file
```

## Getting Started

1. **Run the basic example**: Start with the simple MLP implementation on the basic sequence
2. **Explore model comparison**: Run the advanced section to compare MLP, CNN, and RNN performance
3. **Experiment with parameters**: Try different values for `n_steps`, model architectures, or data generation parameters
4. **Visualize results**: Examine the training loss plots to understand model behavior

## Customization

You can easily modify the code to:
- **Change the lookback window**: Adjust `n_steps` parameter
- **Use different datasets**: Replace the synthetic data generation with your own time series
- **Experiment with architectures**: Modify layer sizes, add dropout, or try different optimizers
- **Add evaluation metrics**: Include additional metrics like MAE, MAPE, or custom loss functions

## Notes

- The models use Mean Squared Error (MSE) as the loss function
- All models are trained for 1000 epochs (you may want to add early stopping for real applications)
- The input data is automatically reshaped to match each model's expected input format
- For the MLP model, 3D input is flattened to 2D, while CNN and RNN models use the full 3D structure

## Contributing

Feel free to contribute by:
- Adding new model architectures (LSTM, GRU, Transformer, etc.)
- Implementing additional evaluation metrics
- Adding support for multivariate time series
- Improving data preprocessing pipelines

## License

This project is available under the MIT License.
