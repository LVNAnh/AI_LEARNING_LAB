from flask import Flask, request, jsonify
import tensorflow as tf 
import numpy as np
from PIL import  Image 
import io

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình MobileNetV2 (đã được huấn luyện trước)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Danh sách nhãn của MobileNetV2
with open("imagenet_labels.txt") as f:
    labels = f.read().splitlines()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Đọc ảnh từ request
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Tiền xử lý ảnh
        img = img.resize((224, 224)) 
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis = 0)

        # Dự đoán với mô hình
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        #Trả kết quả về JSON
        result = [{"label": labels[pred[0]], "confidence": float(pred[2])} for pred in decoded_predictions]
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)