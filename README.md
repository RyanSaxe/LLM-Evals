# LLM-Evals

Repository to explore building custom LLM evaluations.

## Spatial Context Evaluation: Patches

The Needle in a Haystack evaluation shows how models fail to extract information in particular areas of the context as the context size grows. I wanted to design a similar evaluation for spatially constructed prompts. Many LLM use cases leverage structured data like tables and string representations of arrays. This data has clear spatial structure, and we can create evaluations to determine the ways these systems fail at extracting
simple information from those structures such that we can try and improve these systems.

<img alt="Claude Opus" src="llm_evals/spacial/patches/results/claude-3-opus-20240229.png">

This evaluation is structured as follows. Provide the LLM with a grid of characters grouped into patches like so:

[a, a, a, b, b, b, c, c, c]
[a, a, a, b, b, b, c, c, c]
[a, a, a, b, b, b, c, c, c]
[d, d, d, e, e, e, f, f, f]
[d, d, d, e, e, e, f, f, f]
[d, d, d, e, e, e, f, f, f]
[g, g, g, h, h, h, i, i, i]
[g, g, g, h, h, h, i, i, i]
[g, g, g, h, h, h, i, i, i]

Then provide another grid of the same shape. Ask a question about this grid (currently just "what patch is number X in?") in which the answer is definitively a character in the patch. For example, the correct answer to "what patch is number 3 in?" for the grid below is `d`.

[0, 0, 1, 8, 5, 7, 4, 1, 4]
[0, 7, 1, 4, 9, 6, 5, 5, 8]
[2, 6, 8, 6, 5, 1, 2, 9, 5]
[3, 4, 6, 5, 7, 2, 6, 4, 9]
[1, 1, 5, 5, 7, 4, 9, 5, 1]
[1, 9, 1, 9, 2, 0, 2, 6, 4]
[8, 6, 4, 6, 9, 4, 1, 4, 8]
[1, 9, 6, 6, 2, 9, 0, 2, 4]
[5, 5, 1, 2, 0, 6, 4, 6, 6]

## . . . More to Come . . .
