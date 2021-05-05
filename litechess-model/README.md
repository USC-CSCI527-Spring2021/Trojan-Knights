## Litechess Model

This folder contains the code for the following preprocessing tasks:
- Reading PGN files and converting it to hdf5 arrays
- Splitting the created hdf5 arrays into wins for white and wins for black
- Reading the CSV dataset for the regression network

The folder also contains code for the following neural network models:
- Vanilla Autoencoder for Chess Feature Extraction
- Siamese Network for Position Evaluation
- Autoencoder for Missing Pieces Approach (Self-Supervised Learning)
- Regression Network for Evaluation Score Prediction (Self-Supervised Learning)
