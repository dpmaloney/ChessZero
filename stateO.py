#!/usr/bin/env python3
import chess

class State(object):
  def __init__(self, board=None):
    if board is None:
      self.board = chess.Board()
    else:
      self.board = board

  def key(self):
    return (self.board.board_fen(), self.board.turn, self.board.castling_rights, self.board.ep_square)

  def serialize(self):
    import numpy
    x = numpy.zeros(64, dtype=numpy.int8)

    for i in range(64):
      piece = self.board.piece_at(i)
      if piece is not None:
        piece = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                     "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[piece.symbol()]

        x[i] = piece

    x = x.reshape((8,8))
    state = numpy.zeros((2, 8, 8), numpy.uint8)
    state[0] = x
    state[1] = self.board.turn*1.0
    return state

  def edges(self):
    return list(self.board.legal_moves)

if __name__ == "__main__":
  s = State()

s = State()
print(s.serialize())
