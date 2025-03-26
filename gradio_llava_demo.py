import gradio as gr
import base64
import requests
import json

# import gradio as gr
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from lib.utils import  Config
from lib.models import load_model
import numpy as np
import albumentations as A
from lib.dataset import Dataset
from lib import engine
from torchvision import transforms
import requests
from io import BytesIO
from PIL import Image
import cv2


model2d_conf_path = "./conf/convnext-base_14-new.json"
print(f'Loading 2D Model configurations "{model2d_conf_path}"')
config_2d = Config.load_json(model2d_conf_path)
config_2d.model.device = torch.device("cpu")
model_2d = load_model(config_2d)
target_layer_2d = "features"

# model3d_gondo_path = "./checkpoints/pointnet2_8192.t7"
# model_3d_gondo_device = torch.device("cpu")
# model_3d_gondo = PointNet2().to(model_3d_gondo_device)
# model_3d_gondo.load_state_dict(torch.load(model3d_gondo_path, model_3d_gondo_device))

# model_3d_gondo = model_3d_gondo.eval()        

target_list = ['アサ','アズキ','アワ','イネ','エゴマ','オオムギ','カラスザンショウ','キビ','コクゾウムシ','コムギ','ダイズ','ツルマメ','ヒエ']
# target_list_gondo = ['アワ','イネ','エゴマ','カラスザンショウ','コクゾウムシ','コムギ']

name_dic = {'アサ':'Hemp seed',
    'アズキ':'Adzuki bean seed',
    'アワ':'Foxtail millet seed',
    'イネ':'Rice seed',
    'エゴマ':'Perilla seed',
    'オオムギ':'Barley seed',
    'カラスザンショウ':'Japanese Prickly-ash',
    'キビ':'Common millet seed',
    'コクゾウムシ':'Maize weevil',
    'コムギ':'Wheat seed',
    'ダイズ':'Soybean seed',
    'ツルマメ':'Glycine soja seed',
    'ヒエ':'Barnyard millet seed'
    }
    
# SERVER_ADDRESS = "localhost:8000"
# SERVER_ADDRESS = "133.95.105.94:8000"
# llm_server_url = "http://133.95.105.94:8000/image/describe" 
# llm_server_url = "http://localhost:8000/image/describe" 
# llm_server_url = f"http://{SERVER_ADDRESS}/image/describe2" 




params = list(model_2d.parameters())
weight_model2d = np.squeeze(params[-2].data.numpy())

test_aug = A.Compose(
        [
            A.SmallestMaxSize(config_2d.model.size),
            A.CenterCrop(config_2d.model.size[0], config_2d.model.size[1]),
            A.Normalize()
        ]
    )

test_aug2 = A.Compose(
        [
            A.SmallestMaxSize(config_2d.model.size),
            A.CenterCrop(config_2d.model.size[0], config_2d.model.size[1]),
        ]
    )


features_blobs = []

def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

model_2d.base._modules.get(target_layer_2d).register_forward_hook(hook_feature)      
model_2d.eval()  

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# def read_pointcloud(image_path):
#     off_data = o3d.io.read_triangle_mesh(image_path)
#     pointcloud = off_data.sample_points_uniformly(number_of_points=NUMBER_POINTS_GONDO)    
#     pointcloud = np.array(pointcloud.points)
#     pointcloud = pointcloud.astype(np.float32)
#     pointcloud[:, 0:3] = pc_normalize(pointcloud[:, 0:3])
#     pointcloud = torch.tensor(pointcloud)
#     pointcloud = pointcloud.unsqueeze(0)    
#     pred = model_3d_gondo(pointcloud)[0]    
#     pred = pred.cpu()[0]
#     pred = torch.nn.functional.softmax(pred, dim=0).tolist()    
#     confidences = {target_list_gondo[i]: float(pred[i]) for i in range(len(target_list_gondo))}
#     return confidences

# def predict_gondo(image_path):
#     print(f"Got something {image_path}")
#     return read_pointcloud(image_path)

def returnCAM(feature_conv, weight, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []

    for idx in class_idx:
        before_dot = feature_conv.reshape((nc, h * w))
        cam = weight[idx].dot(before_dot)
        cam = cam.reshape(h, w)
        cam = -cam
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam


def explain_prediction(image):
    print("Trying to explain prediction")
    try:
        # Convert Normal Image
        pil_image = Image.fromarray(np.uint8(image))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # Save as JPEG (or change format as needed)
        buffered.seek(0)

        files = {'file': ('image.jpg', buffered, 'image/jpeg')}
        
        print("gonna send to server!")
        response = requests.post(llm_server_url, files=files)
        response.raise_for_status()  # Check for HTTP request errors
            
        # Return the server's response text
        response_text = response.json()["generated_text"]
        return response_text
    except Exception as e:
        return {"error": str(e)}

def generate_heatmap(image):
    if image is None:
        raise gr.Error("画像を入力してください")

    global features_blobs
    try:
        img_filtered = test_aug(image=image)["image"]
        img_filtered2 = test_aug2(image=image)["image"]
    except:
        plt.close()

    img_tensor = torch.from_numpy(img_filtered).permute(2, 0, 1)
    img_var = Variable(img_tensor.unsqueeze(0))
    features_blobs = []
    logit = model_2d.base(img_var)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs, idx = probs.detach().numpy(), idx.numpy()
    cams = returnCAM(features_blobs[0], weight_model2d, [idx[0]])
    graymap = cv2.resize(cams[0], img_filtered2.shape[:2][::-1])
    # graymap = cv2.resize(cams[0], image.shape[:2][::-1])
    # graymap = cams[0]
    heatmap = cv2.applyColorMap(
        graymap, cv2.COLORMAP_JET
    )
    
    result = heatmap * 0.3 + img_filtered2 * 0.5
    result = np.uint8(result)
    features_blobs = []

    return [result,img_filtered2,heatmap]


def predict(image):    
    if image is None:
        raise gr.Error("画像を入力してください")   
    
    # pred = predict_single_image(config_2d, image, model_2d)
        
    image = test_aug(image=image)["image"]
    input = transforms.ToTensor()(image).unsqueeze(0).to(config_2d.model.device)
    
    with torch.no_grad():
        pred, _ = model_2d(input)

    pred = pred.cpu()[0]
    pred = torch.nn.functional.softmax(pred, dim=0).tolist()
    print(f"PRED is {pred}")
    confidences = {target_list[i]: float(pred[i]) for i in range(len(target_list))}

    # confidences = {names[i]: fake_predictions[i] for i in range(len(names))}
    return confidences



# 画像をbase64に変換
def encode_image(image):
    if isinstance(image, np.ndarray):
        # NumPy配列ならPILに変換してBase64エンコード
        pil_image = Image.fromarray(np.uint8(image))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image, str):
        # ファイルパスならそのまま読み込む
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    else:
        raise TypeError("encode_image: Unsupported image type.")
    # with open(image, "rb") as f:
    #     return base64.b64encode(f.read()).decode("utf-8")

# Ollama API に問い合わせる関数
def ask_llava(image1, image2, class_output):
    if image1 is None:
        raise gr.Error("Heatmapボタンを押してください")
    if image2 is None:
        raise gr.Error("Heatmapボタンを押してください")
    if class_output is None:
        raise gr.Error("Predictボタンを押してください")

    encoded_images = [encode_image(image1)]
    encoded_images.append(encode_image(image2))
    
    
    print(list(class_output))

    class_name = name_dic[list(class_output)[0]]

    payload = {
        "model": "llava:7b",
        "prompt": f"Image 1 is an X-ray image, with the target object located in the center. The classification model estimated that this central object is {class_name}. \n\nImage 2 is a heat map highlighting the areas the classification model focused on when identifying objects in image 1. \n\nYour task is to briefly describe the features from these images that may have influenced the classification of the first image as a {class_name}.",
        "images": encoded_images
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                output += data.get("response", "")
        return output

    except Exception as e:
        return f"エラーが発生しました: {str(e)}"



def clear_results():
    return None,"",None,""


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# DRA App")

    with gr.Tab("加藤"):
        with gr.Row():
            with gr.Column():        
                classifier_input = gr.Image(height=355,width=1000)
                predict_button = gr.Button("Predict")   
                heatmap = gr.Button("Heatmap")                 
                explain_button=gr.Button("Explain")        
                clear_button=gr.Button("Clear")                          
            with gr.Column():                    
                classifier_output = gr.Label(num_top_classes=config_2d.model.n_classes)      
                marged_heatmap_image = gr.Image(height=355,width=1000)
                filtered_image = gr.Image(visible=False)
                heatmap_image = gr.Image(visible=False)
                explanation = gr.Textbox(label="説明")
            
            heatmap.click(
                fn=generate_heatmap,
                inputs=classifier_input,
                outputs=[marged_heatmap_image,filtered_image,heatmap_image]
            )
        
            predict_button.click(
                fn=predict, 
                inputs=classifier_input, 
                outputs=classifier_output
            )

            clear_button.click(
                fn=clear_results,
                inputs=None,
                outputs=[classifier_input,classifier_output,marged_heatmap_image,explanation]
            )

            explain_button.click(
                fn=ask_llava,
                inputs=[filtered_image,heatmap_image,classifier_output],
                outputs=explanation
            )

# 実行
if __name__ == "__main__":
    demo.launch()



# def twoguys(image,heatmap):
#     print("Trying to explain prediction")
#     try:
#         # Convert Normal Image
#         pil_image = Image.fromarray(np.uint8(image))
#         buffered_image = BytesIO()
#         pil_image.save(buffered_image, format="JPEG")  # Save as JPEG (or change format as needed)
#         buffered_image.seek(0)

#         # Convert Heatmap
#         pil_heatmap = Image.fromarray(np.uint8(heatmap))
#         buffered_heatmap = BytesIO()
#         pil_heatmap.save(buffered_heatmap, format="JPEG")
#         buffered_heatmap.seek(0)


#         files = {
#             'image': ('image.jpg', buffered_image, 'image/jpeg'),
#             'heatmap': ('heatmap.jpg', buffered_heatmap, 'image/jpeg')
#         }
        
        
#         print("gonna send to server!")
#         response = requests.post(llm_server_url, files=files)
#         response.raise_for_status()  # Check for HTTP request errors
            
#         # Return the server's response text
#         response_text = response.json()["generated_text"]
#         return response_text
#     except Exception as e:
#         return {"error": str(e)}
