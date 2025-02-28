import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# URL của ảnh
img_url = 'https://tropicpet.vn/wp-content/uploads/2024/12/cho-corgi.jpg'

# Tải ảnh từ URL
response = requests.get(img_url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize ảnh về kích thước phù hợp với MobileNetV2

    # Chuyển ảnh thành mảng numpy
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Tải mô hình MobileNetV2
    model = MobileNetV2(weights='imagenet')

    # Dự đoán lớp ảnh
    preds = model.predict(img_array)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    # Hiển thị ảnh
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('D:/output_image.jpg')  # Lưu ảnh vào thư mục D:/output_image.jpg
    plt.show()
else:
    print("Không thể tải ảnh từ URL")
