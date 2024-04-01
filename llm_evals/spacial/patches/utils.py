import numpy as np
from string import alphabet
from typing import Literal

def create_grid(patch_size: int=3, sqrt_patches: Literal[1,2,3,4,5]=3) -> list[list[str]]:
    """create a grid meant to be used as an example label in prompts that
       group sections into labeled square patches where the labels are alphabet characters

    Args:
        patch_size (int, optional): the size of the square patches. Defaults to 3.
        sqrt_patches (int, optional): sqrt_patches ** 2 = number of total patches. Defaults to 3.

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
    grid_size = patch_size * sqrt_patches
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    for i in range(grid_size):
        for j in range(grid_size):
            label = (i // patch_size) * sqrt_patches + (j // patch_size)
            grid[i][j] = alphabet[label]
            
    return grid


def random_index_for_label(label: int, patch_size: int, sqrt_patches: int) -> list[int, int]:
    patch_row = label // sqrt_patches
    patch_col = label % sqrt_patches

    start_row = patch_row * patch_size
    start_col = patch_col * patch_size

    row_offset = np.random.randint(patch_size)
    col_offset = np.random.randint(patch_size)

    final_row = start_row + row_offset
    final_col = start_col + col_offset

    return [final_row, final_col]


def create_grids(n_grid: list[list[int]], number_to_find: int, patch_size: int, sqrt_patches: int) -> dict[str, str]:
    """Create copies of `n_grid` that inserts the number `number_to_find` in inserted once in each patch

    Args:
        n_grid (list[list[int]]): a 2D grid of size (sqrt_patches ** 2, sqrt_patches ** 2) with integers (generally 0-9)
        number_to_find (int): an integer (generally 0-9) to insert to the grid. Grid is expected to exclude this integer.
        patch_size (int): the size of the square patches.
        sqrt_patches (int): sqrt_patches ** 2 = number of total patches.

    Returns:
        dict[str, str]: a mapping from each character to grid where `number_to_find` is in the patch of that key
    """
    outputs = dict()
    for i, label in enumerate(alphabet[:sqrt_patches ** 2]):
        grid = n_grid.copy()
        answer = random_index_for_label(i, patch_size, sqrt_patches)
        grid[*answer] = number_to_find
        outputs[label] = "\n".join([str(row).replace("'","") for row in grid.tolist()])
    return outputs