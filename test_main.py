from flask import Flask, request
from flask_cors import cross_origin

app = Flask(__name__)
import os
import json
import cv2
import numpy

import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

from model.quality_model import efficientnetv2_l as quality_create_model
from model.tessllated_model import resnet34
from unet import Unet
from util.util import name_func, commit_to_mysql, img_file_name
from yolo import YOLO

## 预处理部分
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
quality_data_transform = transforms.Compose(
    [transforms.Resize(512),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# create model
quality_model = quality_create_model(num_classes=4).to(device)
tessellated_model = resnet34(num_classes=2).to(device)

# load model weights
quality_model_weight_path = "./weights/quality_l/model-99.pth"
quality_model.load_state_dict(torch.load(quality_model_weight_path, map_location=device))
weights_path = "./weights/resNet34.pth"
tessellated_model.load_state_dict(torch.load(weights_path, map_location=device))
yolo = YOLO()
unet = Unet()


## 预测函数

# 异常图像识别
def quality_classification(model, img):
    # read class_indict
    json_path = './class_indices/quality_class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    img = quality_data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    for i in range(len(predict)):
        if i == predict_cla:
            print("该图片类别为:", class_indict[str(i)])
    return predict_cla


# 豹纹状改变分类
def tessellated_classification(model, img):
    data_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(256),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    saturation = ImageEnhance.Color(img)
    img = saturation.enhance(4)
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.5)
    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(img.to(device)).cpu()
        predict_y = torch.max(output, dim=1)[1]
        predict = predict_y[0].item()
    return predict


# 病灶区域分割算法
def unet_predict(img, new_name, content, filename):
    img = unet.detect_image(img, filename)
    img.save(new_name)
    img.save("D:/eye/highmyopiafront/public" + content[5])
    commit_to_mysql(content[0], content[1], new_name, content[2], content[3], content[4], content[5])


# 小目标病灶检测算法
def yolo_predict(img, new_name_2, content):
    r_image, have_weiss = yolo.detect_image(img)
    # r_image.save(new_name_2)
    # r_image.save("D:/eye/highmyopiafront/public" + content[6])
    commit_to_mysql(content[0], content[1], new_name_2, content[2], content[3], content[4], content[6])
    return have_weiss


@app.route('/detect', methods=['GET'])
@cross_origin()
def index():
    img_path = request.args.get('localpath')
    case_id = request.args.get('case_id')
    exam_id = request.args.get('exam_id')
    path = request.args.get('path')
    type = request.args.get('type')
    downfile = request.args.get('downfile')
    dev = request.args.get('dev')
    # img_path_2 = img_path[:len(img_path) - 4] + '-1.jpg'

    # img_path = img_path[1:]
    new_name, new_sec_name, new_name_2, new_sec_name_2 = name_func(img_path, path)
    file_name = img_file_name(img_path)

    content = [case_id, exam_id, type, downfile, dev, new_sec_name, new_sec_name_2]
    res, res_lan = mainFunc(img_path, new_name, content, file_name, new_name_2)
    return res_lan


def mainFunc(img_path, new_name, content, file_name, new_name_2):
    print(img_path)
    img = Image.open(img_path)
    quality_cls = quality_classification(quality_model, img)
    res = '普通近视'
    res_lan = ''
    if quality_cls == 2:
        print("图片质量正常，下一步进行豹纹状改变检测--")
        res_lan += "图片质量正常\n"
        tessllated_cls = tessellated_classification(tessellated_model, img)
        if tessllated_cls == 1:
            print("检测到清晰豹纹状改变")
            res_lan += ",检测到清晰豹纹状改变\n"
            res = '高度近视'
            print("下一步进行病灶区域分割")
            unet_predict(img, new_name, content, file_name)
            print("病灶区域如图")
            res_lan += ",病灶区域如图\n"
            print("进行小目标病灶检测")
            have_weiss = yolo_predict(img, new_name_2, content)
            if have_weiss:
                res_lan += "检测到玻璃体浑浊病变\n"
            else:
                res_lan += "未检测到小目标病灶\n"
        else:
            print("无明显豹纹状改变")
            res_lan += ",无明显豹纹状改变\n"
    elif quality_cls == 0:
        print("图片检测失败！")
        print("图片质量异常，类型为睫毛遮挡")
        print("请重新输入图片")
        res_lan += "图片检测失败!图片质量异常，类型为睫毛遮挡,请重新输入图片\n"
    elif quality_cls == 1:
        print("图片检测失败！")
        print("图片质量异常，类型为眼睑遮挡")
        print("请重新输入图片")
        res_lan += "图片检测失败!图片质量异常，类型为眼睑遮挡,请重新输入图片\n"
    else:
        print("图片检测失败！")
        print("图片质量异常，拍摄位置不规范造成伪影")
        print("请重新输入图片")
        res_lan += "图片检测失败!图片质量异常，拍摄位置不规范造成伪影,请重新输入图片\n"
    return res, res_lan


# 主函数
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(port=4091, host="127.0.0.1", debug=True)
