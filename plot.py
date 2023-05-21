#!/usr/bin/env python3

# @Author: Huseyin Tutan 110190021

import matplotlib.pyplot as plt

methods = ['DecisionTree', 'K_Nearest', 'Linear_Regression', 'Neural_Networks', 'RandomForest', 'SVMachine', 'XGBoost']
scores = [0.20099, 0.22402, 0.47510, 0.4629, 0.16422, 0.20317, 0.16357]

# En düşük değere sahip olan sütunu belirle
min_score = min(scores)
min_index = scores.index(min_score)

# Grafik için sütunları çiz
plt.bar(methods, scores, color=['green' if i == min_index else 'blue' for i in range(len(methods))])

# Diğer grafik ayarlamaları
plt.xlabel('Methods')
plt.ylabel('Scores')
plt.title('Comparison of Scores')
plt.xticks(rotation=45)
plt.tight_layout()

# Grafiği göster
plt.show()
