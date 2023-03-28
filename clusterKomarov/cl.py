import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

df = pd.read_csv('blackjack.csv', sep=',')  # Кредиты банка
df.drop(df.columns[[3]], axis=1)  # удаляем последний столбец классов
# plotting dendrogram # Linkage Matrix
plt.figure(figsize=(12, 7))
dendrogram(linkage(df, method='ward'))  # ,color_threshold = 3
plt.title('Дендограмма Комарова ')
plt.ylabel('Euclidean distance')
plt.xlabel('клиенты банка (параметры)')
plt.show()