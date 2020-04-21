#!/usr/bin/env python3
from __future__ import print_function
import os
import chess
import time
import chess.svg
import traceback
import base64
from state import State

class Valuator(object):
  def __init__(self):
    import torch
    from train import Net
    vals = torch.load("nets/value.pth")
    self.model = Net()
    self.model.load_state_dict(vals)
    self.reset()
    self.memo = {}

  def __call__(self, s):
    
    return self.value(s)

  def value(self, s):
    import torch
    brd = s.serialize()[None]
    print(brd)
    output = self.model(torch.tensor(brd).float())
    print(output.data)
    return float(output.data[0][0])

  def reset(self):
      self.count = 0



v = Valuator()

board = State()
print(v(board))

board.board.push_san("e4")
print(board.board)
print(v(board))

board.board.push_san("d5")
print(board.board)
print(v(board))
