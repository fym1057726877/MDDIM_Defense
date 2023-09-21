from utils import get_project_path
import torch


class Au(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__ == '__main__':
    w = Au(2, 3)
    print(w.a)
