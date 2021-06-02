import pygame as pg
import os
import random
import torch
from snake import SnakeNet

WIDTH = 1000
HEIGHT = 1000
PIXEL_WIDTH = 50
PIXEL_HEIGHT = 50
NUM_X_POINTS = int(WIDTH/PIXEL_WIDTH)
NUM_Y_POINTS = int(WIDTH/PIXEL_HEIGHT)


class Food(pg.sprite.Sprite):
    def __init__(self, x_coordinate=None, y_coordinate=None):
        super().__init__()
        if x_coordinate is None:
            x_coordinate = random.randint(0, NUM_X_POINTS - 1)
        if y_coordinate is None:
            y_coordinate = random.randint(0, NUM_Y_POINTS - 1)
        self.image = pg.image.load('./food.png')
        self.rect = self.image.get_rect()
        self.x = x_coordinate
        self.y = y_coordinate
        self.rect.center = [self.x*50 + 25, self.y*50 + 25]

    def new_pos(self):
        self.x = random.randint(0, NUM_X_POINTS - 1)
        self.y = random.randint(0, NUM_Y_POINTS - 1)
        self.rect.center = [self.x*50 + 25, self.y*50 + 25]

    def print(self):
        print(f'<| Food at indx {self.x}, {self.y}')


class Body(pg.sprite.Sprite):
    def __init__(self, x_coordinate, y_coordinate, head=False):
        super().__init__()
        if head:
            self.image = pg.image.load('./head.png')
        else:
            self.image = pg.image.load('./body.png')
        self.rect = self.image.get_rect()
        self.x = x_coordinate
        self.y = y_coordinate
        self.rect.center = [self.x*50 + 25, self.y*50 + 25]


class Snake:
    def __init__(self, name='Sir Finley'):
        self.tail = [[4, 4], [3, 4]]
        self.length = 1
        self.score = 0
        self.x_dir = 1
        self.y_dir = 0
        self.name = name

    def move_up(self):
        self.x_dir = 0
        self.y_dir = -1

    def move_down(self):
        self.x_dir = 0
        self.y_dir = 1

    def move_left(self):
        self.x_dir = -1
        self.y_dir = 0

    def move_right(self):
        self.x_dir = 1
        self.y_dir = 0

    def update_pos(self):
        self.tail.insert(0, self.tail.pop())
        self.tail[0][0] = self.tail[1][0] + self.x_dir
        self.tail[0][1] = self.tail[1][1]+ self.y_dir

    def got_food(self):
        self.score += 1
        if self.x_dir == 0 and self.y_dir == -1:
            self.tail.append([self.tail[-1][0], self.tail[-1][1] + 1])
        if self.x_dir == 0 and self.y_dir == 1:
            self.tail.append([self.tail[-1][0], self.tail[-1][1] - 1])
        if self.x_dir == 1 and self.y_dir == 0:
            self.tail.append([self.tail[-1][0] - 1, self.tail[-1][1]])
        if self.x_dir == -1 and self.y_dir == 0:
            self.tail.append([self.tail[-1][0] + 1, self.tail[-1][1]])

    def check_limit(self):
        return 0 <= self.tail[0][0] <= NUM_X_POINTS - 1 and 0 <= self.tail[0][1] <= NUM_Y_POINTS - 1

    def print_tail(self):
        board = [['-' for _ in range(NUM_X_POINTS)] for _ in range(NUM_Y_POINTS)]
        for idx, body in enumerate(self.tail):
            if idx == 0:
                board[body[1]][body[0]] = 'H'
            else:
                board[body[1]][body[0]] = 'B'
        for row in board:
            s = ''
            for c in row:
                s += '  ' + c
            print(s)

    def collision_tail(self):
        head_x = self.tail[0][0]
        head_y = self.tail[0][1]
        for idx in range(1, len(self.tail)):
            if head_x == self.tail[idx][0] and head_y == self.tail[idx][1]:
                return True
        return False

    def print(self):
        print(f'I am {self.name}! I have a tail of length {self.length} and {self.score} score.')


def draw_screen(screen, x_coordinate=0, y_coordinate=0):
    background_img = pg.image.load('./grid.png')
    screen.blit(background_img, [x_coordinate, y_coordinate])


def update_sprite_group(sprite_group, snake, food=None):
    sprite_group.empty()
    sprite_group.add(food)
    for idx, tail_square in enumerate(snake.tail):
        if idx == 0:
            sprite_group.add(Body(tail_square[0], tail_square[1], head=True))
        else:
            sprite_group.add(Body(tail_square[0], tail_square[1]))
    return sprite_group


def collision_food(snake, food):
    return snake.tail[0][0] == food.x and snake.tail[0][1] == food.y


def main_game():
    pg.init()
    clock = pg.time.Clock()
    pg_screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption('pysnake')
    icon = pg.image.load('./anaconda.png')
    pg.display.set_icon(icon)

    piece_group = pg.sprite.Group()

    snakE = Snake()
    snakE.print()
    food = Food()

    piece_group = update_sprite_group(piece_group, snakE, food)
    running_game = True
    snakenet = SnakeNet()
    weights = torch.load('./snake_318.pth', map_location=lambda storage, loc: storage)
    snakenet.load_state_dict(weights)
    while running_game:
        if not snakE.check_limit():
            running_game = False
            continue
        for event in pg.event.get():
            """
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_w:
                    print('<| Changing direction to up :: x=0, y=-1')
                    if snakE.y_dir != 1:
                        snakE.move_up()
                if event.key == pg.K_s:
                    print('<| Changing direction to up :: x=0, y=1')
                    if snakE.y_dir != -1:
                        snakE.move_down()
                if event.key == pg.K_a:
                    print('<| Changing direction to up :: x=-1, y=0')
                    if snakE.x_dir != 1:
                        snakE.move_left()
                if event.key == pg.K_d:
                    print('<| Changing direction to up :: x=1, y=0')
                    if snakE.x_dir != -1:
                        snakE.move_right()
            """
            if event.type == pg.QUIT:
                running_game = False
        # Get agent action


        snakE.update_pos()
        if collision_food(snakE, food):
            food.new_pos()
            snakE.got_food()
        if snakE.collision_tail():
            running_game = False
        piece_group = update_sprite_group(piece_group, snakE, food)
        draw_screen(pg_screen)
        piece_group.draw(pg_screen)
        pg.display.update()
        clock.tick(4)
    print('<| Game over => terminating...')
    print(f'<| Your score was: {snakE.score}')


if __name__ == '__main__':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    main_game()
