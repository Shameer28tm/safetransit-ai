import matplotlib.pyplot as plt
import numpy as np

# Class names
classes = ['car', 'accident']

# Metrics for each class (from your result)
precision = [0.996, 0.972]
recall = [0.996, 0.999]
map50 = [0.995, 0.995]
map5095 = [0.904, 0.860]

x = np.arange(len(classes))
width = 0.2

# Plot
plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, precision, width, label='Precision', color='skyblue')
plt.bar(x - 0.5*width, recall, width, label='Recall', color='lightgreen')
plt.bar(x + 0.5*width, map50, width, label='mAP@0.5', color='salmon')
plt.bar(x + 1.5*width, map5095, width, label='mAP@0.5:0.95', color='orange')

plt.xlabel('Classes')
plt.ylabel('Score')
plt.ylim(0.8, 1.05)
plt.title('Model Evaluation Metrics by Class')
plt.xticks(x, classes)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
