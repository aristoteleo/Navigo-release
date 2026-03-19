import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from adjustText import adjust_text

df = pd.read_csv('Atrial#cardiomyocytes_Ventricular#cardiomyocytes_deg.csv')
log2fc = df['logfoldchanges'].clip(-20, 20)
pvals = -np.log10(df['pvals_adj'])
plt.rcParams['font.family'] = 'Liberation Sans'

fig, ax = plt.subplots(figsize=(16, 8))

threshold_p = -np.log10(0.05)
threshold_fc = 1

atrial = (df['logfoldchanges'] > threshold_fc) & (pvals > threshold_p)
ventricular = (df['logfoldchanges'] < -threshold_fc) & (pvals > threshold_p)

colors = plt.cm.tab20.colors
red = colors[6]
blue = colors[0]

ax.scatter(log2fc[~(atrial | ventricular)], pvals[~(atrial | ventricular)], 
           c='gray', alpha=0.5, s=20, zorder=1)
ax.scatter(log2fc[atrial], pvals[atrial], c=red, alpha=0.6, s=20, label='Atrial', zorder=2)
ax.scatter(log2fc[ventricular], pvals[ventricular], c=blue, alpha=0.6, s=20, label='Ventricular', zorder=2)

ax.axhline(threshold_p, color='black', linestyle='--', linewidth=0.8)
ax.axvline(threshold_fc, color='black', linestyle='--', linewidth=0.8)
ax.axvline(-threshold_fc, color='black', linestyle='--', linewidth=0.8)

texts = []
top_atrial = df[atrial].nlargest(30, 'scores')
for _, row in top_atrial.iterrows():
    x_pos = min(row['logfoldchanges'], 20)
    texts.append(ax.text(x_pos, -np.log10(row['pvals_adj']+1e-300), row['names'], 
                        fontsize=15, fontstyle='italic'))

top_ventricular = df[ventricular].nsmallest(30, 'scores')
for _, row in top_ventricular.iterrows():
    x_pos = max(row['logfoldchanges'], -20)
    texts.append(ax.text(x_pos, -np.log10(row['pvals_adj']+1e-300), row['names'], 
                        fontsize=15, fontstyle='italic'))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))

ax.set_xlim(-22, 22)
ax.set_xlabel('Log2 Fold Change', fontsize=20)
ax.set_ylabel('-Log10 P-value', fontsize=20)
ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig('volcano_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()