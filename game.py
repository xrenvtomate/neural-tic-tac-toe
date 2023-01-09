class Game:
    def __init__(self):
        self.field = [[0] * 3 for _ in range(3)]

    def place(self, p, x, y):
        if not (0 <= x < 3 and 0 <= y < 3):
            print('not correct positon')
            return False
        if self.field[y][x] != 0:
            print('this positon is already taken')
            return False

        self.field[y][x] = p
        return True

    def check_cells(self, *cells):
        mult = 1
        for el in cells:
            mult *= self.field[el[0]][el[1]]
        if mult == 1:
            return 1
        if mult == 8:
            return 2
        return 0

    def __str__(self):
        res = ''
        for row in self.field:
            res += ' '.join(
                map(lambda x: 'x' if x == 1 else ('.' if x == 0 else 'o'), row)
            ) + '\n'

        return res

    def check_game(self):
        res = 0
        for x in range(3):
            if res:
                break
            res = self.check_cells((x, 0), (x, 1), (x, 2))
        for y in range(3):
            if res:
                break
            res = self.check_cells((0, y), (1, y), (2, y))

            if res == 0:
                res = self.check_cells((0, 0), (1, 1), (2, 2))
            if res == 0:
                res = self.check_cells((2, 0), (1, 1), (0, 2))

        if all([all(row) for row in self.field]):
            res = 3
        return res
