import numpy as np
import random
from matplotlib import pyplot as plt

from game import Game
from nnconfig import hidden_neurons as h_n, epochs, test_games, learning_rate


# initialize neural net
h = np.zeros(h_n)
w_h = np.random.uniform(-0.5, 0.5, (h_n, 18))
b_h = np.zeros((h_n, 1))
o = np.zeros(9)
w_o = np.random.uniform(-0.5, 0.5, (9, h_n))
b_o = np.zeros((9, 1))


def field_to_1d(field):
    field_1d = np.array(field).ravel()
    res = []
    for el in field_1d:
        if el == 1:
            res.append(1)
        else:
            res.append(0)
    for el in field_1d:
        if el == 2:
            res.append(1)
        else:
            res.append(0)
    return res  


def forward_pass(inp):
    inp = np.array(inp)
    inp.shape += (1,)
    h = w_h @ inp + b_h
    h = 1 / (1 + np.exp(-h))
    o = w_o @ h + b_o
    o = 1 / (1 + np.exp(-o))
    return o


def error_backprop(inp, chosen, is_right):
    inp = np.array(inp)
    inp.shape += (1,)

    global w_h, b_h, w_o, b_o

    h = w_h @ inp + b_h
    h = 1 / (1 + np.exp(-h))
    o = w_o @ h + b_o
    o = 1 / (1 + np.exp(-o))

    if is_right:
        right = np.zeros((9, 1))
        right[chosen, 0] = 1
    else:
        right = np.ones((9, 1))
        right[chosen, 0] = 0

    delta_o = right - o
    w_o += learning_rate * delta_o @ np.transpose(h)
    b_o += learning_rate * delta_o

    delta_h = np.transpose(w_o) @ -delta_o * (h * (1 - h))
    w_h -= learning_rate * delta_h @ np.transpose(inp)
    b_h -= learning_rate * delta_h
    

x_epochs = []
y_wins = []

for epoch in range(1, epochs + 1):
    moves = []
    game = Game()
    res = 0
    p = p_start = random.randint(1, 2)
    while res == 0:
        field_1d = field_to_1d(game.field)
        preds = forward_pass(field_1d)
        free_cells = filter(lambda x: game.field[x//3][x%3] == 0, range(9))
        if p == 1:
            chosen_cell = max(free_cells, key=lambda x: preds[x])  # neural net move
        else:
            chosen_cell = random.choice(list(free_cells))  # monke move
        x, y = chosen_cell % 3, chosen_cell // 3
        if p == 1:
            moves.append((field_1d, chosen_cell))

        game.place(p, x, y)
        res = game.check_game()
        p = 3 - p

    is_well_played = (p_start == 2 and res == 3) or res == 1
        
    for case in moves:
        error_backprop(*case, is_well_played)


    # test
    if epoch % 1000 == 0:
        # print('epoch: ', epoch)
        win = draw = loss = 0
        for _ in range(test_games):
            game = Game()
            res = 0
            p = random.randint(1, 2)
            while res == 0:
                field_1d = field_to_1d(game.field)
                preds = forward_pass(field_1d)
                if p == 1:
                    best = max(filter(lambda x: game.field[x//3][x%3] == 0, range(9)), key=lambda x: preds[x])
                else:
                    best = random.choice(list(filter(lambda x: game.field[x//3][x%3] == 0, range(9))))
                x, y = best % 3, best // 3

                game.place(p, x, y)
                res = game.check_game()
                p = 3 - p
            if res == 1:
                win += 1
            elif res == 2:
                loss += 1
            else:
                draw += 1

        x_epochs.append(epoch)
        y_wins.append(win / test_games)
        # print(f'results after {test_games} test games with monke:')
        # print(f'win: {100 * win // test_games}%')
        # print(f'loss: {100 * loss // test_games}%')
        # print(f'draw: {100 * draw // test_games}%')

plt.plot(x_epochs, y_wins)
plt.ylim((0, 1))
plt.show()