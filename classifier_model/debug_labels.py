import pandas as pd
df = pd.read_parquet('src/data/cleaned_train.parquet')
df['bias'] = df['bias'].astype(str).str.strip().str.title()
bias_map = {'0': 'Left', '1': 'Center', '2': 'Right', '0.0': 'Left', '1.0': 'Center', '2.0': 'Right'}
df['bias'] = df['bias'].replace(bias_map)

with open("debug_out_py.txt", "w", encoding="utf-8") as f:
    f.write('--- LEFT SAMPLES ---\n')
    for t in df[df['bias']=='Left']['content'].tolist()[:5]:
        f.write(t[:200] + '\n')

    f.write('\n--- RIGHT SAMPLES ---\n')
    for t in df[df['bias']=='Right']['content'].tolist()[:5]:
        f.write(t[:200] + '\n')

