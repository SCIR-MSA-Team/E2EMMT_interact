import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt


def visual_text(tokens, scores, save_path):
    import matplotlib.colors as colors
    cmap = cm.get_cmap('BuGn')

    off = (sum(scores) / len(scores)) * 1.0
    normer = colors.Normalize(vmin=min(scores)-off, vmax=max(scores)+off)
    colors = [colors.to_hex(cmap(normer(x))) for x in scores]

    if len(tokens) != len(colors):
        raise ValueError("number of tokens and colors don't match")

    style_elems = []
    span_elems= []
    for i in range(len(tokens)):
        style_elems.append(f'.c{i} {{ background-color: {colors[i]}; }}')
        span_elems.append(f'<span class="c{i}">{tokens[i]} </span>')

    with open(save_path,'w') as file:
        file.write(f"""<!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet"><style>span {{ font-family: "Roboto Mono", monospace; font-size: 12px; }} {' '.join(style_elems)}</style></head><body>{' '.join(span_elems)}</body></html>""")


def read_file(path, cast=False):
    res = [
        line.strip()
        for line in open(path).read().split('\n')
        if len(line.strip()) > 0
    ]
    if cast:
        res = [float(line) for line in res]
    return res


# parser = argparse.ArgumentParser()
# parser.add_argument('--text_path', type=str, default='text')
# parser.add_argument('--score_path', type=str, default='score')
# parser.add_argument('--cmap', type=str, default='BuGn')
# parser.add_argument('--alpha', type=float, default=1.0)
# args = parser.parse_args()

# tokens = ["wow" , "this","movie"]
# scores = [0.036130528076202865,0.0017988330714969398,0.004889735250681301]

# visual_text(tokens, scores, './visual/index.html')