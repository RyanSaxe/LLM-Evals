from string import ascii_lowercase as alphabet
from typing import Literal

import numpy as np


def create_grid(patch_size: int = 3, n_patches_per_row: Literal[1, 2, 3, 4, 5] = 3) -> list[list[str]]:
    """create a grid meant to be used as an example label in prompts that
       group sections into labeled square patches where the labels are alphabet characters

    Args:
        patch_size (int, optional): the size of the square patches. Defaults to 3.
        n_patches_per_row (int, optional): n_patches_per_row ** 2 = number of total patches. Defaults to 3.

    Returns:
        list[list[str]]: a 2D grid that labels patches

    Example:

        >>> create_grid(3, 3)

            [['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            ['d', 'd', 'd', 'e', 'e', 'e', 'f', 'f', 'f'],
            ['d', 'd', 'd', 'e', 'e', 'e', 'f', 'f', 'f'],
            ['d', 'd', 'd', 'e', 'e', 'e', 'f', 'f', 'f'],
            ['g', 'g', 'g', 'h', 'h', 'h', 'i', 'i', 'i'],
            ['g', 'g', 'g', 'h', 'h', 'h', 'i', 'i', 'i'],
            ['g', 'g', 'g', 'h', 'h', 'h', 'i', 'i', 'i']]
    """
    grid_size = patch_size * n_patches_per_row
    grid = [["" for _ in range(grid_size)] for _ in range(grid_size)]

    for i in range(grid_size):
        for j in range(grid_size):
            label = (i // patch_size) * n_patches_per_row + (j // patch_size)
            grid[i][j] = alphabet[label]

    return grid


def random_index_for_label(
    label: int, patch_size: int, n_patches_per_row: int, rng: np.random.Generator
) -> tuple[int, int]:
    patch_row = label // n_patches_per_row
    patch_col = label % n_patches_per_row

    start_row = patch_row * patch_size
    start_col = patch_col * patch_size

    row_offset = rng.integers(patch_size)
    col_offset = rng.integers(patch_size)

    random_row = start_row + row_offset
    random_col = start_col + col_offset

    return random_row, random_col


def create_grids(
    n_grid: np.ndarray,
    number_to_find: int,
    patch_size: int,
    n_patches_per_row: int,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Create copies of `n_grid` that inserts the number `number_to_find` in inserted once in each patch

    Args:
        n_grid (np.ndarray): a 2D grid of size (n_patches_per_row ** 2, n_patches_per_row ** 2) with ints (atm 0-9)
        number_to_find (int): an integer (atm 0-9) to insert to the grid. Grid is expected to exclude this integer.
        patch_size (int): the size of the square patches.
        n_patches_per_row (int): n_patches_per_row ** 2 = number of total patches.

    Returns:
        dict[str, str]: a mapping from each character to grid where `number_to_find` is in the patch of that key
    """
    outputs = dict()
    for i, label in enumerate(alphabet[: n_patches_per_row**2]):
        grid = n_grid.copy()
        answer = random_index_for_label(i, patch_size, n_patches_per_row, rng)
        grid[*answer] = number_to_find
        outputs[label] = "\n".join([str(row).replace("'", "") for row in grid.tolist()])
    return outputs
