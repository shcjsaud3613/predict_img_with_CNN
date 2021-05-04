# -*- coding: utf-8 -*-
from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

image_w = 64
image_h = 64

pixels = image_h * image_w * 3


categories = ['갈비구이', '갈치구이', '고등어구이', '곱창구이', '닭갈비', '떡갈비', '불고기', '삼겹살', '장어구이', '조개구이', '조기구이', '황태구이', '훈제오리', '계란국', '떡국_만두국', '무국', '미역국', '북엇국', '육개장', '콩나물국', '과메기', '양 념치킨', '젓갈', '콩자반', '편육', '피자', '후라이드치킨', '경단', '꿀떡', '송편', '만두', '라면', '막국수', '물냉면', '비빔냉면', '수제비', '열무국수', '잔치국수', '짜장면', '짬뽕', '쫄면', '칼국수', '콩국수', '김밥', '김치볶음밥', '비빔밥', '새우볶음밥', '알밥', '유부초밥', '두부김치', '떡볶이', '라볶이', '멸치볶음', '소세지볶음', '어묵볶음', '오징어채볶음', '제육볶음', '주꾸미볶음', '보쌈', '간장게장', '양념게장', '떡꼬치', '감자전', '김치전', '파전', '곱창전골', '전복죽', '김치찌개', '닭계장', '동태찌개', '된장찌개', '순두부찌개', '갈비찜', '계란찜', '김치찜', '꼬막찜', '닭볶음탕', '수육', '순대', '족발', '찜닭', '해물찜', '갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕', '고추튀김', '새우튀 김', '오징어튀김', '멍게', '산낙지', '물회', '육회']


def load_my_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on Google Colab)
    global model
    model = load_model('./model/multi_img_classification.model')
    print(" * Keras model loading is success")


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    response = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
            image = image.resize((image_w, image_h))
            data = np.asarray(image)

            X = []
            X.append(data)

            X = np.array(X)

            prediction = model.predict(X)
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

            pre_ans = prediction[0].argmax()
            print(pre_ans)

            pre_ans_str = categories[pre_ans]
            print(pre_ans_str)

            response["predictions"] = pre_ans_str
            response["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(response)

# 메인함수
if __name__ == '__main__':
    print((" * Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_my_model()
    app.run(host='0.0.0.0')