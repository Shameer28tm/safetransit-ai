import matplotlib.pyplot as plt

# Overall model performance (from your output)
metrics = {
    'Precision': 0.985,
    'Recall': 0.994,
    'mAP@0.5': 0.9936,
    'mAP@0.5:0.95': 0.7037
}

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'salmon', 'orange'])

plt.title('Overall Model Performance')
plt.ylabel('Score')
plt.ylim(0.8, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
