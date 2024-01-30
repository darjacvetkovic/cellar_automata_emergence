import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


seeds = {
    "diehard": [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 1],
    ],
    "boat": [[1, 1, 0], [1, 0, 1], [0, 1, 0]],
    "r_pentomino": [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
    "pentadecathlon": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "beacon": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
    "acorn": [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]],
    "spaceship": [[0, 0, 1, 1, 0], [1, 1, 0, 1, 1], [1, 1, 1, 1, 0], [0, 1, 1, 0, 0]],
    "block_switch_engine": [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ],
    "infinite": [
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1],
    ],
}


def survival(x, y, universe):
    """
    Compute one iteration of Life for one cell.
    :param x: x coordinate of cell in the universe
    :type x: int
    :param y: y coordinate of cell in the universe
    :type y: int
    :param universe: the universe of cells
    :type universe: np.ndarray
    """
    num_neighbours = np.sum(universe[x - 1 : x + 2, y - 1 : y + 2]) - universe[x, y]
    # The rules of Life
    if universe[x, y] and not 2 <= num_neighbours <= 3:
        return 0
    elif num_neighbours == 3:
        return 1
    return universe[x, y]


def run_GoL_one_generation(universe):
    """
    Compute one iteration of Life for the universe.
    :param universe: initial universe of cells
    :type universe: np.ndarray
    :return: updated universe of cells
    :rtype: np.ndarray
    """
    new_universe = np.copy(universe)
    # Apply the survival function to every cell in the universe
    for i in range(universe.shape[0]):
        for j in range(universe.shape[1]):
            new_universe[i, j] = survival(i, j, universe)
    return new_universe


def animate_life(universe_size, seed, seed_position, quality=200, cmap="binary", n_generations=50, interval=300, save=False, animate=False):
    """
    Animate the Game of Life.
    :param universe_size: dimensions of the universe
    :type universe_size: tuple (int, int)
    :param seed: initial starting array
    :type seed: list of lists, np.ndarray
    :param seed_position: coordinates where the top-left corner of the seed array should be pinned
    :type seed_position: tuple (int, int)
    :param cmap: the matplotlib cmap that should be used
    :param n_generations: number of universe iterations, defaults to 30
    :param n_generations: int, optional
    :param interval: time interval between updates (milliseconds), defaults to 300ms
    :param interval: int, optional
    :param save: whether the animation should be saved, defaults to False
    :param save: bool, optional
    """
    # Initialise the universe and seed
    universe = np.zeros(universe_size, dtype=np.float32)
    x_start, y_start = seed_position[0], seed_position[1]
    seed_array = np.array(seeds[seed])
    x_end, y_end = x_start + seed_array.shape[0], y_start + seed_array.shape[1]
    universe[x_start:x_end, y_start:y_end] = seed_array
    initial_universe = universe.copy()

    dynamics = []
    dynamics.append(universe)
    for _ in range(n_generations):
        universe = run_GoL_one_generation(universe)
        dynamics.append(universe)

    return np.array(dynamics, dtype=np.float32), initial_universe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyGameofLife. By default, produces 50 generations of the 'infinite' seed")
    parser.add_argument("--universe-size", type=str, default="20,20", help="comma-separated dimensions of universe (x by y)")
    parser.add_argument(
        "-seed",
        type=str,
        default="spaceship",
        help="diehard, boat, spaceship, infinite, block_switch_engine, beacon, acorn, pentadecathlon, r_pentomino",
    )
    parser.add_argument("-n", type=int, default=30, help="number of universe iterations")
    parser.add_argument("-quality", type=int, default=100, help="image quality in DPI")
    parser.add_argument("-cmap", type=str, default="binary", help="colour scheme")
    parser.add_argument("-interval", type=int, default=300, help="interval (in milliseconds) between iterations")
    parser.add_argument(
        "--seed-position",
        type=str,
        default="9,9",
        help="comma-separated coordinates of seed",
    )
    parser.add_argument("--animate", type=bool, default=True, help="")

    args = parser.parse_args()

    animate_life(
        universe_size=(int(args.universe_size.split(",")[0]), int(args.universe_size.split(",")[1])),
        seed=args.seed,
        quality=args.quality,
        cmap=args.cmap,
        seed_position=(int(args.seed_position.split(",")[0]), int(args.seed_position.split(",")[1])),
        n_generations=args.n,
        interval=args.interval,
        save=True,
        animate=args.animate,
    )
