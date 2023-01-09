import random
import sys

import numpy as np

from game import Game
from utils import field_to_1d, forward_pass

dump_folder = 'dumps'
if len(sys.argv) == 2:
    dump_folder = sys.argv[1]

w_o = np.load(f'{dump_folder}/w_o.npy')
b_o = np.load(f'{dump_folder}/b_o.npy')
w_h = np.load(f'{dump_folder}/w_h.npy')
b_h = np.load(f'{dump_folder}/b_h.npy')

nn_params = [w_h, b_h, w_o, b_o]


while True:
    game = Game()
    print(game)
    p = random.randint(1, 2)

    res = 0
    while res == 0:
        field_1d = field_to_1d(game.field)
        preds = forward_pass(field_1d, *nn_params)
        free_cells = filter(lambda x: game.field[x // 3][x % 3] == 0, range(9))
        if p == 1:
            chosen_cell = max(free_cells, key=lambda x: preds[x])
            x, y = chosen_cell % 3, chosen_cell // 3
        else:
            x, y = map(int, input().split())

        game.place(p, x, y)
        print(game)
        res = game.check_game()
        p = 3 - p
