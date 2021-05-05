import chess
import chess.pgn
import chess.engine
import random
import pickle
import argparse
import sys
import numpy as np
import itertools as it
import gzip
import tables as tb
import time
import csv

def read_games(csv_fname):
  with open(csv_fname, "r") as games:
    for game in csv.reader(games):
      yield game


def bit_string_gen(a):

    ''' 
    
    Maps the given position of a piece into a bit string of 64 dimensional 
    representing the position of the piece type on a 64 cell chess board 
    
    '''
    
    bit_string = list( map(lambda x: 1 if x in a else 0, range(0, 64)) )
    
    return bit_string

def bit_board_gen(given_position, baseboard, eval_val):
    
    ''' 
    
    Input : 1) Given_position - a Board() object of the present position
            2) Baseboard  - a BaseBoard() object of the present position
            3) Winner  - Eventual winner of the given position 
    
    Output : A 'bit-board-representation' of the board.
    
    Turns a position into a 773 dimensional vector:
    
    773 = number of colours * number of pieces-types * size of chess board + additional board information
        = (2 * 6 * 64) + 5
 
    '''

    bit_string = []
    for color in [True,False]:
        for piece in range(1,7):
            
            indices = list(baseboard.pieces(piece, color))
            bit_string.extend(bit_string_gen(indices))
     

    # Checking the side to move. 1 for white and 0 for black
    bit_string.extend(list(map(int, [given_position.turn])))

    # Checking castling rights of white and balck
    bit_string.extend(list( map( int,
                                      [ 
        given_position.has_kingside_castling_rights(True),
        given_position.has_queenside_castling_rights(True),
        given_position.has_kingside_castling_rights(False),
        given_position.has_queenside_castling_rights(False)  
            ] ) ) )
    
    bit_string.append(eval_val)
    
    bit_string = np.asarray(bit_string, dtype=np.int32)

    return bit_string


def bit_board_array(game, eval_val):

    bit_list = []
    state = chess.Board(game)

    # FEN represntation of the board
    position = state.board_fen() 

    # A board representing the position of chess pieces.
    baseboard = chess.BaseBoard(position) 

    bit_list.append( bit_board_gen( state, baseboard, eval_val ) )

    bit_list = np.asarray(bit_list, dtype=np.int32)
    return bit_list

#######################################################################    

def main():
    '''
    Input - A PGN file name 
    
    Output - A hdf5 file 
    '''

    csv_fname = "/home/parthsuresh/DeepChess/data/chessData.csv"
    games_generator = read_games(csv_fname)
    next(games_generator)

    games_file = tb.open_file('data/games_reg.h5', mode='w')
    games_count = 0
    games_created = False

    while(True):
        try:
            game, eval_val = next(games_generator)
            
            games_count += 1

            if (games_count % 10000 == 0):         
                print(games_count)
            
            try:
                if eval_val.startswith("#"):
                    flag = True
                    eval_val = int(eval_val[1:])
                    if eval_val < 0:
                        eval_val = -10000 - eval_val
                    else:
                        eval_val = 10000 - eval_val
                else:
                    eval_val = int(eval_val)
            except:
                print(eval_val)
                continue

            if not games_created:
                game_example = bit_board_array(game, eval_val)
                games_arr = games_file.create_earray( games_file.root, "games", obj=game_example )
                games_created = True
            games_arr.append(bit_board_array(game, eval_val))
            
        except Exception as e:
            print("Exception")
            print(e)
            break

    games_file.close()

#######################################################################
        
        
main()

