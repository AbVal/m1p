import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


def get_all_mutants(X):
    ret = []
    for p1, p2, p3 in permutations(X):
        ret.append(p1 + 0.5 * (p2 - p3))
    return np.array(ret)


def get_all_mutants_noisy(X, coef=0.33):
    ret = []
    for p1, p2, p3 in permutations(X):
        ret.append(p1 + 0.5 * (p2 - p3) + coef * np.random.rand(len(p1)))
    return np.array(ret)


def get_all_crossovers(x, y):
    return [x, y, np.array([x[0], y[1]]), np.array([y[0], x[1]])]


def visualize_basic():
    X = np.array([[-1, 0], [0, 1], [1, 0]])

    mut = get_all_mutants(X)

    crossovers = []

    for x in X:
        for y in mut:
            crossovers += get_all_crossovers(x, y)

    crossovers = np.array(crossovers)
    plt.scatter(X[:, 0], X[:, 1], s=80, label='Кандидаты')
    plt.scatter(mut[:, 0], mut[:, 1], s=80, color='red', label='Мутанты')

    plt.scatter(crossovers[:, 0], crossovers[:, 1], s=20, color='black', label='Новое поколение')
    plt.legend()
    plt.title('Все возможные точки нового поколения')
    plt.savefig('../figures/V.pdf', bbox_inches='tight')


def visualize_noisy():
    X = np.array([[-1, 0], [0, 1], [1, 0]])

    mut = get_all_mutants_noisy(X)

    crossovers = []

    for x in X:
        for y in mut:
            crossovers += get_all_crossovers(x, y)

    crossovers = np.array(crossovers)
    plt.scatter(X[:, 0], X[:, 1], s=80, label='Кандидаты')
    plt.scatter(mut[:, 0], mut[:, 1], s=80, color='red', label='Мутанты')

    plt.scatter(crossovers[:, 0], crossovers[:, 1], s=20, color='black', label='Новое поколение')
    plt.legend()
    plt.title('Все возможные точки нового поколения после зашумления')
    plt.savefig('../figures/V_noisy.pdf', bbox_inches='tight')
