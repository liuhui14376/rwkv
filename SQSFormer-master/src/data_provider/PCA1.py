from data_provider.data_provider import train_loader
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

import os
import spectral

# 加载高光谱图像
img = train_loader
#img = spectral.open_image('path_to_your_hyperspectral_image.hdr')

# 获取图像数据
data = img.load()

# 将三维图像数据转换为二维矩阵
# 其中，每一行代表一个像素，每一列代表一个波段
pixels = data.reshape(-1, data.shape[-1])

# 设置要保留的主成分数量
n_components = 10  # 例如，保留前10个主成分

# 初始化PCA对象
pca = PCA(n_components=n_components)

# 对像素数据进行PCA拟合和转换
pca_data = pca.fit_transform(pixels)

# 将PCA处理后的数据转换回类似原始图像的格式
# 注意：这里我们得到的是一个降维后的图像数组，而不是严格意义上的高光谱图像
pca_reshaped = pca_data.reshape(data.shape[0], data.shape[1], n_components)

# 如果需要，可以将降维后的数据保存为NumPy数组或图像文件
np.save('pca_processed_hyperspectral_data.npy', pca_reshaped)

# 如果你想查看某个主成分，可以将其转换为灰度图像并保存
# 这里以第一个主成分为例
first_component = pca_reshaped[:, :, 0]
first_component_img = Image.fromarray(np.uint8(first_component * 255))  # 将数据缩放到0-255范围并转换为图像
first_component_img.save('first_pca_component.png')

# 如果需要，也可以可视化PCA处理后的某个波段

plt.imshow(first_component, cmap='gray')
plt.title('First PCA Component')
plt.colorbar()
plt.show()
