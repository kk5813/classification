from predict import quality_classification
from flask import Flask

# app=Flask(__name__)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # app.run(port=4091, host="127.0.0.1", debug=True)
    while True:
        img_path = input('请输入图片路径:')
        # ./plot_img/011253-20201121@105551-L2-S.jpg
        # D:\eyeproject\show\plot_img\001128-20190823@102828-R1-S.jpg
        quality_cls = quality_classification(img_path)
        if quality_cls == 2:
            print("图片质量正常，下一步进行豹纹状改变检测--")
        elif quality_cls == 0:
            print("图片异常，类型为睫毛遮挡，请重新输入图片")
        elif quality_cls == 1:
            print("图片异常，类型为眼睑遮挡，请重新输入图片")
        else:
            print("图片异常，拍摄位置不规范造成伪影，请重新输入图片")
