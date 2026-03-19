import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

import pandas as pd
import numpy as np

# Get method names in order
methods = []
method_counts = {}
for col in df.columns[1:]:
    method = col.split('.')[0]
    if method not in method_counts:
        methods.append(method)
        method_counts[method] = 0
    method_counts[method] += 1

# Collect all non-NaN values per method
method_values = {method: [] for method in methods}
for col in df.columns[1:]:
    method = col.split('.')[0]
    method_values[method].extend(df[col].dropna().tolist())

# Build new df with extended columns
new_df = pd.DataFrame()
new_df['Unnamed: 0'] = df['Unnamed: 0']

for method in methods:
    total_vals = len(method_values[method])
    for i in range(total_vals):
        col_name = f'{method}.{i}' if i > 0 else method
        if col_name in df.columns:
            new_df[col_name] = df[col_name]
        else:
            new_df[col_name] = np.nan

# Add overall row
overall_dict = {'Unnamed: 0': 'overall'}
for method in methods:
    for i, val in enumerate(method_values[method]):
        col_name = f'{method}.{i}' if i > 0 else method
        overall_dict[col_name] = val

overall_df = pd.DataFrame([overall_dict])
df_extended = pd.concat([new_df, overall_df], ignore_index=True)

df_extended.to_csv("extended_data.csv")

breakpoint()