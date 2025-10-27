import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from solution_guidance.cslib import fetch_ts

def run_eda(data_dir='cs-train', outdir='reports'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ts_data = fetch_ts(data_dir)
    df_all = ts_data['all']

    plt.figure(figsize=(10,5))
    df_all.groupby('year_month')['revenue'].sum().plot(kind='bar', color='teal')
    plt.title('Monthly Revenue (All Countries)')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(f'{outdir}/monthly_revenue_all.png')

    top_countries = list(ts_data.keys())[:10]
    plt.figure(figsize=(10,6))
    for country in top_countries:
        df_country = ts_data[country]
        plt.plot(df_country['date'], df_country['revenue'], label=country)
    plt.legend()
    plt.title('Daily Revenue per Country')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(f'{outdir}/daily_revenue_countries.png')

    corr = df_all[['purchases','unique_invoices','unique_streams','total_views','revenue']].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{outdir}/feature_correlation.png')

    print(f"EDA reports saved to {outdir}")
