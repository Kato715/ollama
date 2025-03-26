import base64
import requests
import json

# 複数の画像ファイルパス
image_paths = ['./pic/asa_086.jpg', './pic/asa_086_heatmap.jpg']
class_name = "Hemp seed"

# base64 エンコードされた画像リストを作成
encoded_images = []
for path in image_paths:
    with open(path, 'rb') as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
        encoded_images.append(encoded)

# プロンプト（1つの質問に対して複数画像を渡せる）
payload = {
    "model": "llava:7b",
    # "prompt": f"Image 1 is the X-ray image, where the target object is located in the center. The classification model has estimated that the central object is {class_name}. \n\nImage 2 is a heat map highlighting the areas the classification model focused on when identifying objects in the image 1. \n\nYour task is to briefly describe the features from these images that may have influenced the classification of the image 1 as an {class_name}.",
    "prompt": f"Image 1 is an X-ray image, with the target object located in the center. The classification model estimated that this central object is {class_name}. \n\nImage 2 is a heat map highlighting the areas the classification model focused on when identifying objects in image 1. \n\nYour task is to briefly describe the features from these images that may have influenced the classification of the first image as a {class_name}.",
    "images": encoded_images
}

response = requests.post('http://localhost:11434/api/generate', json=payload, stream=True)

# 返答を連結して1つの文章にまとめる
output_text = ""
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8'))
        output_text += data.get("response", "")

print(output_text)

