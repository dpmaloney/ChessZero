#!/usr/bin/env python3
import numpy as np
from tensorflow import keras
import wandb
wandb.init(project="chesslearning")

class ChessValueDataset():
  def __init__(self):
    dat = np.load("processed/dataset_25M.npz")
    self.X = dat['arr_0']
    self.Y = dat['arr_1']
    print("loaded", self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Input((2, 8, 8)))
    model.add(keras.layers.Conv2D(64, kernel_size=2, padding="same"))
    model.add(keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(keras.layers.Conv2D(256, kernel_size=4, padding="same"))
    model.add(keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(loss="hinge", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
  device = "cuda"

  chess_dataset = ChessValueDataset()
  print(np.shape(chess_dataset.X))
  print(chess_dataset.Y[0])
  model = build_model()
  model.fit(chess_dataset.X, chess_dataset.Y, epochs=5, batch_size=128)
  model.save("model.h5")



