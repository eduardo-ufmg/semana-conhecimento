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
    # sanitize metric name: avoid '_' that breaks legend
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

# --- aesthetics ---
models = sorted(long['model'].unique())
metrics = sorted(long['metric'].unique())

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
metric_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(metrics)}

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
model_styles = {mod: {'linestyle': linestyles[i % len(linestyles)],
                      'marker': markers[i % len(markers)],
                      'markersize': 7}
                for i, mod in enumerate(models)}

# --- plotting ---
datasets = long['dataset'].unique().tolist()
x = np.arange(len(datasets))

fig, ax = plt.subplots(figsize=(20, 8))
half_width = 0.25

for ds_idx, ds in enumerate(datasets):
    subset = long[long['dataset'] == ds]
    for _, row in subset.iterrows():
        model = row['model']
        metric = row['metric']
        val = float(row['value'])

        if val < 0.5:
            val = np.nan

        xpos = ds_idx
        ax.plot([xpos - half_width, xpos + half_width], [val, val],
                linestyle=model_styles[model]['linestyle'],
                linewidth=2.2,
                color=metric_colors[metric], zorder=5)
        ax.plot(xpos, val,
                marker=model_styles[model]['marker'],
                markersize=model_styles[model]['markersize'],
                linestyle='None',
                markeredgewidth=0.9,
                markeredgecolor='black',
                markerfacecolor=metric_colors[metric],
                zorder=6)

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_xlim(-0.6, len(datasets) - 0.4)
ax.set_ylabel('Acurácia Alcançada')
ax.set_title('Comparação entre Modelos e Métricas de Otimização')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# --- explicit legend handles (sanitized labels) ---
metric_handles = [Line2D([0], [0], color=metric_colors[m], lw=3) for m in metrics]
metric_labels  = [str(m) if str(m) != '' else '0' for m in metrics]

model_handles = [Line2D([0], [0],
                        color='black',
                        linestyle=model_styles[m]['linestyle'],
                        marker=model_styles[m]['marker'],
                        markersize=7,
                        markeredgecolor='black',
                        markerfacecolor='white')
                 for m in models]
model_labels = [str(m) for m in models]

# Place legends to the right, but make room for them so they are not clipped:
# adjust the subplot area so right space is available for the legends
plt.subplots_adjust(right=3.0)   # leave space on the right for legends

leg_metrics = ax.legend(metric_handles, metric_labels, title='Métrica Otimizada (cor)',
                        loc='upper left', bbox_to_anchor=(1.02, 1))
ax.add_artist(leg_metrics)

leg_models = ax.legend(model_handles, model_labels, title='Modelo (marcador)',
                       loc='upper left', bbox_to_anchor=(1.02, 0.45))
# note: we added metric legend as artist first, then the model legend

plt.tight_layout()

# --- saving ---
plt.savefig('results_plot.png')
plt.show()
