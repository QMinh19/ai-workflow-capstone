import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def evaluate_model_performance(true_values, rf_preds, prophet_preds, outdir='reports'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    mae_rf = mean_absolute_error(true_values, rf_preds)
    mae_prophet = mean_absolute_error(true_values, prophet_preds)
    rmse_rf = np.sqrt(mean_squared_error(true_values, rf_preds))
    rmse_prophet = np.sqrt(mean_squared_error(true_values, prophet_preds))

    plt.figure(figsize=(10,6))
    plt.plot(true_values, label='True', color='black')
    plt.plot(rf_preds, label=f'RF (MAE={mae_rf:.2f})', color='royalblue')
    plt.plot(prophet_preds, label=f'Prophet (MAE={mae_prophet:.2f})', color='orange')
    plt.legend()
    plt.title('Model Comparison: RandomForest vs Prophet')
    plt.xlabel('Time Step')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(f"{outdir}/model_comparison.png")

    print(f"MAE_RF={mae_rf:.3f}, MAE_Prophet={mae_prophet:.3f}")
    print(f"RMSE_RF={rmse_rf:.3f}, RMSE_Prophet={rmse_prophet:.3f}")

    return {
        "mae_rf": mae_rf,
        "mae_prophet": mae_prophet,
        "rmse_rf": rmse_rf,
        "rmse_prophet": rmse_prophet
    }