import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Tạo dữ liệu mảng 2 chiều (heatmap data)
data = np.random.rand(82, 82)

# Tạo heatmap sử dụng seaborn
sns.heatmap(data, cmap='viridis')  # 'viridis' là một trong các colormap có sẵn trong matplotlib
plt.savefig("heatmap.png")
# Hiển thị heatmap
plt.show()
