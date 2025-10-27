import os
from solution_guidance.cslib import fetch_ts

def load_all_data(train_dir='cs-train', prod_dir='cs-production'):
    data = {}
    for label, ddir in [('train', train_dir), ('production', prod_dir)]:
        if not os.path.isdir(ddir):
            raise FileNotFoundError(f"{ddir} not found")
        print(f"Loading {label} data from {ddir}...")
        data[label] = fetch_ts(ddir)
    return data


if __name__ == '__main__':
    all_data = load_all_data()
    print(f"Loaded datasets: {list(all_data.keys())}")