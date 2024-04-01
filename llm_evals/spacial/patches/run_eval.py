import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_image
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from llm_evals.models.anthropic import AnthropicModel
from llm_evals.models.openai import OpenAIModel
from llm_evals.spacial.patches.utils import alphabet, create_grid, create_grids


def precision(row):
    score = (row.index == row.values).astype(float)
    counts = row.value_counts().to_dict()
    for i, key in enumerate(row.index):
        score[i] /= counts.get(key, 1)
    return pd.Series(score, index=row.index)


def viz_grid(grid_data: list[list[int]]) -> go.Figure:
    color_grid = []
    for row in grid_data:
        colored_row = [[0, 0, 0] if color < 0 else [(1 - color) * 255, color * 255, 0] for color in row]
        color_grid.append(colored_row)
    fig = go.Figure(go.Image(z=color_grid))
    return fig.update_layout(
        hovermode=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )


def run_eval(args):
    rng = np.random.default_rng(args.seed)
    if "claude" in args.model:
        llm = AnthropicModel(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    else:
        llm = OpenAIModel(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    all_prompts = []
    all_results = []
    for patch_size in range(args.patch_size_start, args.patch_size_end + 1):
        total_hits = args.n_patches_per_row**2 * args.n_samples
        progress = tqdm(total=total_hits)
        sample_results = []
        sample_prompts = []
        for _ in range(args.n_samples):
            grid_shape = [patch_size * args.n_patches_per_row] * 2
            s_grid = create_grid(patch_size, args.n_patches_per_row)
            s_grid_for_prompt = "\n".join([str(row).replace("'", "") for row in s_grid])
            choices = [x for x in range(10) if x != args.number_to_find]
            n_grid = rng.choice(choices, size=grid_shape)
            outputs = create_grids(n_grid, args.number_to_find, patch_size, args.n_patches_per_row, rng)
            system_prompt = (
                'You are an expert in Visual Spatial reasoning. Below is a square grid broken up into "patches": sections labeled with the same letter.'
                "\n\n"
                f"{s_grid_for_prompt}"
                "\n\n"
                f"Your goal is, given a grid of the same size containing numbers, to determine which patch the number {args.number_to_find} in the grid belongs to."
                "\n\n"
                "Please respond only with the single letter of the corresponding patch and nothing else."
            )
            results = dict()
            prompts = dict()
            for label, n_grid_for_prompt in outputs.items():
                results[label] = llm(n_grid_for_prompt, system_prompt)
                prompts[label] = dict(system_prompt=system_prompt, prompt=n_grid_for_prompt)
                progress.update(1)
            sample_results.append(results)
            sample_prompts.append(prompts)
        progress.close()
        all_results.append(sample_results)
        all_prompts.append(sample_prompts)

    scores = {k: [] for k in all_results[0][0].keys()}
    for result in all_results:
        # score = pd.DataFrame(result).apply(lambda col: col == col.name).mean().to_dict() # accuracy
        score = pd.DataFrame(result).apply(precision, axis=1).mean().to_dict()
        for key, value in score.items():
            scores[key].append(value)

    obj_to_save = dict(results=all_results, args=vars(args), prompts=all_prompts, scores=scores)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_folder) / f"{args.model}.json", "w") as f:
        json.dump(obj_to_save, f)

    figures = []
    for i in range(args.n_patches_per_row):
        for j in range(args.n_patches_per_row):
            label = alphabet[i * args.n_patches_per_row + j]
            fig = viz_grid([scores[label]] * len(scores[label]))
            figures.append(fig)

    fig = make_subplots(
        rows=args.n_patches_per_row,
        cols=args.n_patches_per_row,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    for i, single_fig in enumerate(figures):
        for trace in single_fig.data:
            fig.add_trace(trace, row=i // args.n_patches_per_row + 1, col=i % args.n_patches_per_row + 1)

    for axis in fig.layout:
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            fig.layout[axis].tickmode = "array"
            fig.layout[axis].tickvals = []
            fig.layout[axis].showticklabels = False
    fig.update_layout(
        height=400,
        width=400,
        hovermode=False,
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    fig.update_layout(
        plot_bgcolor="rgb(0, 0, 0)",
        paper_bgcolor="rgb(0, 0, 0)",
    )
    write_image(fig, Path(args.output_folder) / f"{args.model}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--number_to_find",
        type=int,
        default=3,
        help="the number to inject into the prompt that the LLM has to find",
    )
    parser.add_argument(
        "--patch_size_start",
        type=int,
        default=1,
        help="min size for each patch in the generated grid for the prompt",
    )
    parser.add_argument(
        "--patch_size_end", type=int, default=8, help="max size for each patch in the generated grid for the prompt"
    )
    parser.add_argument(
        "--n_patches_per_row",
        type=Literal[1, 2, 3, 4, 5],
        default=3,
        help="max size for each patch in the generated grid for the prompt",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="How many prompts to generate for each patch_size x patch"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-opus-20240229",
        help="name of the model to eval. currently only anthropic and openai supported.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="name of the model to eval. currently only anthropic and openai supported.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature hyperparameter",
    )
    parser.add_argument(
        "--max_tokens",
        type=float,
        default=100,  # can be very small. Optimal response for this task is 1 token per query.
        help="max tokens the LLM can have in the output",
    )
    params, unparsed = parser.parse_known_args()
    run_eval(params)


if __name__ == "__main__":
    main()
