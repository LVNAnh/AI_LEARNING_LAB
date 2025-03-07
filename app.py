from flask import Flask, request, jsonify 
import pytesseract
import numpy as np
import cv2
from PIL import Image
import io

#Khởi tạo flask app
app = Flask(__name__)

#Cấu hình đường dẫn đến Tesseract OCR ()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#Hàm tiền xử lý ảnh
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #chuyển sang ảnh grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Nhị phân hóa ảnh
    return thresh

#Endpoint xử lý nhận dạng văn bản
@app.route("/ocr", methods=["POST"])
def recognize_text():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())) # Đọc ảnh từ file upload
    image = np.array(image) #Chuyển ảnh thành mảng numpy

    #Tiền xử lý ảnh
    processed_image = preprocess_image(image)

    #Nhận dạng văn bản bằng Tesseract
    text = pytesseract.image_to_string(processed_image)

    return jsonify({"recognized text": text})

if __name__ == "__main__":
    app.run(debug=True)