import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- load & preprocess ---
raw = pd.read_csv('aggregated_results.csv', header=None)
col_models = raw.iloc[1].fillna('').astype(str).tolist()
col_metrics = raw.iloc[2].fillna('').astype(str).tolist()
data = raw.iloc[3:].copy().reset_index(drop=True)

new_cols = ['dataset']
for i in range(1, raw.shape[1]):
    m = col_models[i].strip()
    met = col_metrics[i].strip()
    if m in ('', '_'):
        # propagate previous model name
        for j in range(i-1, -1, -1):
            if col_models[j].strip() not in ('', '_'):
                m = col_models[j].strip(); break
    # sanitize metric name: avoid '_' that breaks automatic legend behavior
    if met in ('', '_'):
        met = '0'
    new_cols.append((m, met))
data.columns = new_cols

# Flatten to long format
flat = {'dataset': data['dataset'].astype(str)}
for i in range(1, data.shape[1]):
    model, metric = data.columns[i]
    flat[f"{model}||{metric}"] = data.iloc[:, i].astype(str)
df = pd.DataFrame(flat)
long = df.melt(id_vars=['dataset'], var_name='model_metric', value_name='value')
split = long['model_metric'].str.split(r'\|\|', expand=True)
long[['model', 'metric']] = split
long['value'] = pd.to_numeric(long['value'].str.replace('[^0-9.+-eE]', '', regex=True), errors='coerce')
long = long.dropna(subset=['value']).reset_index(drop=True)
long['dataset'] = long['dataset'].str.strip()

# --- USER-TUNABLE AESTHETIC PARAMETERS ---
FIGSIZE = (18, 7)
SPACING_FACTOR = 0.78       # <1 compresses distances between dataset sections (use 0.6..1.0)
HALF_WIDTH = 0.25           # half-length of top-edge line in x units (will be scaled)
TITLE_FS = 20
LABEL_FS = 18
TICK_FS = 12
LEGEND_FS = 14
LEGEND_HANDLE_LENGTH = 3.2
LEGEND_MARKERSCALE = 1.6
LEGEND_LABEL_SPACING = 0.35   # smaller keeps entries dense
LEGEND_BORDERPAD = 0.9        # enlarge legend box internal padding
# -------------------------------------------------------

# --- aesthetics ---
models = sorted(long['model'].unique())
metrics = sorted(long['metric'].unique())

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
metric_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(metrics)}

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
model_styles = {mod: {'linestyle': linestyles[i % len(linestyles)],
                      'marker': markers[i % len(markers)],
                      'markersize': 8}
                for i, mod in enumerate(models)}

# --- plotting ---
datasets = long['dataset'].unique().tolist()
# compress spacing (use SPACING_FACTOR to control density)
x_base = np.arange(len(datasets))
x = x_base * SPACING_FACTOR

fig, ax = plt.subplots(figsize=FIGSIZE)

# scale HALF_WIDTH with spacing so the top-edge length looks consistent after compression
half_width = HALF_WIDTH * SPACING_FACTOR

for ds_idx, ds in enumerate(datasets):
    subset = long[long['dataset'] == ds]
    for _, row in subset.iterrows():
        model = row['model']
        metric = row['metric']
        val = float(row['value'])

        # If value is below 0.5, treat as missing (for accuracy metrics)
        if val < 0.5:
            val = np.nan

        xpos = x[ds_idx]
        ax.plot([xpos - half_width, xpos + half_width], [val, val],
                linestyle=model_styles[model]['linestyle'],
                linewidth=2.4,
                color=metric_colors[metric],
                zorder=5)
        ax.plot(xpos, val,
                marker=model_styles[model]['marker'],
                markersize=model_styles[model]['markersize'],
                linestyle='None',
                markeredgewidth=0.9,
                markeredgecolor='black',
                markerfacecolor=metric_colors[metric],
                zorder=6)

# Axis formatting: ticks, labels, title
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=TICK_FS)
ax.set_xlim(x.min() - (0.6 * SPACING_FACTOR), x.max() + (0.4 * SPACING_FACTOR))
ax.set_ylabel('Acurácia Alcançada', fontsize=LABEL_FS)
ax.set_title('Comparação entre Modelos e Métricas de Otimização', fontsize=TITLE_FS)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# --- explicit legend handles ---
metric_handles = [Line2D([0], [0], color=metric_colors[m], lw=3) for m in metrics]
metric_labels = [str(m) for m in metrics]

model_handles = [Line2D([0], [0],
                        color='black',
                        linestyle=model_styles[m]['linestyle'],
                        marker=model_styles[m]['marker'],
                        markersize=model_styles[m]['markersize'],
                        markeredgecolor='black',
                        markerfacecolor='white')
                 for m in models]
model_labels = [str(m) for m in models]

# Make room on the right for larger legend boxes
plt.subplots_adjust(right=0.78)   # adjust this if you need more/less room

# Metric legend: larger box, compact entries
leg_metrics = ax.legend(metric_handles, metric_labels,
                        title='Métrica Otimizada (cor)',
                        loc='upper left',
                        bbox_to_anchor=(1.01, 1.0),
                        prop={'size': LEGEND_FS},
                        handlelength=LEGEND_HANDLE_LENGTH,
                        labelspacing=LEGEND_LABEL_SPACING,
                        borderpad=LEGEND_BORDERPAD,
                        markerscale=LEGEND_MARKERSCALE,
                        frameon=True)
ax.add_artist(leg_metrics)

# Model legend placed under the metric legend
leg_models = ax.legend(model_handles, model_labels,
                       title='Modelo (marcador / estilo)',
                       loc='upper left',
                       bbox_to_anchor=(1.01, 0.45),
                       prop={'size': LEGEND_FS},
                       handlelength=LEGEND_HANDLE_LENGTH,
                       labelspacing=LEGEND_LABEL_SPACING,
                       borderpad=LEGEND_BORDERPAD,
                       markerscale=LEGEND_MARKERSCALE,
                       frameon=True)

# Increase tick label font size (y axis)
ax.tick_params(axis='y', labelsize=TICK_FS)

plt.tight_layout()
# Save using bbox_inches='tight' to ensure legends are not clipped when saved
plt.savefig('results_plot.png', bbox_inches='tight', dpi=1200)
plt.show()
