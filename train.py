import numpy as np
import random
from matplotlib import pyplot as plt

from game import Game
from nnconfig import hidden_neurons as h_n, epochs
from utils import field_to_1d, forward_pass, error_backprop

# initialize neural net
h = np.zeros(h_n)
w_h = np.random.uniform(-0.5, 0.5, (h_n, 18))
b_h = np.zeros((h_n, 1))
o = np.zeros(9)
w_o = np.random.uniform(-0.5, 0.5, (9, h_n))
b_o = np.zeros((9, 1))
nn_params = [w_h, b_h, w_o, b_o]


x_epochs = []
y_wins = []

wins = draw = loss = 0
for epoch in range(1, epochs + 1):
    moves = []
    game = Game()
    res = 0
    p = p_start = random.randint(1, 2)
    while res == 0:
        field_1d = field_to_1d(game.field)
        preds = forward_pass(field_1d, *nn_params)
        free_cells = filter(lambda x: game.field[x // 3][x % 3] == 0, range(9))
        if p == 1:
            chosen_cell = max(free_cells, key=lambda x: preds[x])
        else:
            chosen_cell = random.choice(list(free_cells))
        x, y = chosen_cell % 3, chosen_cell // 3
        if p == 1:
            moves.append((field_1d, chosen_cell))

        game.place(p, x, y)
        res = game.check_game()
        p = 3 - p

    is_well_played = (res == 3) or res == 1
    for case in moves:
        error_backprop(*case, is_well_played, *nn_params)
    if not is_well_played:
        for _ in range(7):
            error_backprop(*moves[-1], is_well_played, *nn_params)

    if epoch % 10000 == 0:
        print('epoch: ', epoch)
        print(f'wins: {100 * wins // 10000}%')
        print(f'loss: {100 * loss // 10000}%')
        print(f'draw: {100 * draw // 10000}%')
        print(f'losses: {loss}')
        x_epochs.append(epoch)
        y_wins.append(wins / 10000)

        wins = draw = loss = 0

    if res == 1:
        wins += 1
    elif res == 2:
        loss += 1
    else:
        draw += 1

plt.plot(x_epochs, y_wins)
plt.ylim((0, 1))
plt.show()
ans = input('wanna save model? "y" to save: ')
if ans == 'y':
    np.save('dumps/w_o.npy', w_o)
    np.save('dumps/b_o.npy', b_o)
    np.save('dumps/w_h.npy', w_h)
    np.save('dumps/b_h.npy', b_h)
