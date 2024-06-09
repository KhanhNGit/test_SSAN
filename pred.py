from retina_face import *
import cv2
import math
import numpy as np
from torchvision import transforms
import onnxruntime as ort

app = FaceAnalysis()
app.prepare()

def crop_face_from_scene(image, box, scale=1.3):
    y1,x1,w,h=box
    y2=y1+w
    x2=x1+h
    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=np.int16(max(math.floor(y1),0))
    x1=np.int16(max(math.floor(x1),0))
    y2=np.int16(min(math.floor(y2),w_img))
    x2=np.int16(min(math.floor(x2),h_img))
    region=image[x1:x2,y1:y2]
    return region[:,:,::-1]

def crop_face(img_path):
    try:
        img = cv2.imread(img_path)
        faces = app.get(img)
        box = faces[0][0], faces[0][1], faces[0][2]-faces[0][0], faces[0][3]-faces[0][1]
        if box[2] < 30 or box[3] < 30:
            return None
        return crop_face_from_scene(image=img, box=box)
    except:
        return None

def transform(image):
    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformed_image = img_transforms(image)
    return transformed_image

model_path = '/weights/best_model.onnx'
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

img_path = input()

if crop_face(img_path) == None:
    print("The input image is corrupted. Please try another image")
else:
    image = crop_face(img_path)
    transformed_image = transform(image)
    input_tensor = transformed_image.reshape(input_shape).cpu().numpy()
    outputs = sess.run(None, {input_name: input_tensor})
    print(outputs)
    # output_s = outputs[0].flatten()
    # if(class_names[np.argmax(output_s)][:4] == i):
