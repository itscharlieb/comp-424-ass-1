import numpy as np
import math
import csv
import pprint
from collections import defaultdict


PRINTER = pprint.PrettyPrinter(indent=4)


def target_func(x):
    return np.cos(x**2 / 2) / np.log(x+2)


def successors(x, step):
    return filter(lambda y: y > 0 and y < 10, [x+step, x-step])


def hill_climb(x_0, step):
    y_0 = target_func(x_0)

    iterations = 0
    while(True):
        sx = successors(x_0, step)
        sxy = [(x, target_func(x)) for x in sx]
        x_next, y_next = max(sxy, key=lambda xy: xy[1])

        if y_next > y_0:
            x_0, y_0 = x_next, y_next
            iterations += 1
        else:
            return x_0, y_0, iterations


def boltzmann(x_0, x_1, temp):
    return math.e ** ((x_1 - x_0) / temp)


def simulated_annealing(x_0, step):
    y_0 = target_func(x_0)
    x_max, y_max = x_0, y_0

    iterations = 0
    stable_iterations = 0
    temp = 1
    while(stable_iterations < 10):
        iterations += 1

        sx = successors(x_0, step)
        sxy = [(x, target_func(x)) for x in sx]
        x_next, y_next = max(sxy, key=lambda xy: xy[1])

        # update best if next values are better
        if y_next > y_max:
            x_max, y_max = x_next, y_next

        # search x_next if y_next is better than y_max
        if y_next > y_0:
            x_0, y_0 = x_next, y_next

        #search x_next probabilistically even if y_next worse than y_max
        elif np.random.rand() < boltzmann(y_0, y_next, temp / iterations):
            x_0, y_0 = x_next, y_next

        # no change to x_0, increment number of stable iterations
        else:
            stable_iterations += 1

    return x_max, y_max, iterations


def run_hill_climb():
    results = []
    for x_start in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for step in [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10]:
            x_max, y_max, iterations = hill_climb(x_start, step)
            results.append([x_start, step, x_max, y_max, iterations])
    return results


def best_results(results):
    d = defaultdict(list)

    # convert to dictionary of {start: (step, x_max, y_max, iterations)}
    for r in results:
        d[r[0]].append((r[1], r[2], r[3], r[4]))

    # sort and take two best
    for _, v in d.items():
        # sort by y_max
        v.sort(key=lambda r: -r[2])
        del v[2:]

    # return as list in same format
    return [
        (start, y[0], y[1], y[2], y[3])
        for start, x in d.items()
        for y in x
    ]


def write_results(results, fname):
    with open(fname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["X Start", "Step Size", "X Max", "Y Max", "Num Iterations"])
        writer.writerows(results)


# run computations
results = run_hill_climb()
write_results(results, "data/hillclimb.csv")

best = best_results(results)
write_results(best, "data/besthillclimb.csv")

anneals = []
for start, step, _, _, _ in best:
    x_max, y_max, iterations = simulated_annealing(start, step)
    anneals.append((start, step, x_max, y_max, iterations))
write_results(anneals, "data/anneal.csv")
