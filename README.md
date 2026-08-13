# ReLU-KAN_For_Time_Series_On_FinalDataNN
This model is designed to predict the output of a type of electrical circuit.

This repository contains the implementation of a ReLU-based Kolmogorov-Arnold Network (ReLU-KAN) designed specifically for Time Series Forecasting. The model is applied to predict the dynamic output behavior of a specific type of electrical circuit using the `FinalDataNN` dataset.

 📖 Project Overview

Kolmogorov-Arnold Networks (KANs) have emerged as a powerful alternative to traditional Multi-Layer Perceptrons (MLPs). While standard KANs often use smooth B-splines as activation functions, this project explores ReLU-KAN, which utilizes Rectified Linear Units to efficiently capture non-linear temporal dependencies in time-series data.

Predicting the output of electrical circuits (such as voltage or current over time) requires a model that can handle sequential data and complex non-linear dynamics. ReLU-KAN provides a highly efficient and mathematically sound architecture for this engineering task.

🎯 Objectives
- Process sequential time-series data from electrical circuit simulations/measurements.
- Implement and train a ReLU-KAN architecture to map input signals/states to circuit outputs.
- Evaluate the model's accuracy in forecasting continuous time-series values.

📂 Repository Structure

```text
ReLU-KAN_For_Time_Series_On_FinalDataNN/
├── FinalDataNN.xlsx                     # Dataset containing circuit time-series data
├── ReLU_KAN_TS_for_FinalDataNN.ipynb    # Jupyter Notebook for step-by-step training and visualization
├── relu_kan_ts_for_finaldatann.py       # Standalone Python script for production training/inference
└── README.md                            # Project documentation
