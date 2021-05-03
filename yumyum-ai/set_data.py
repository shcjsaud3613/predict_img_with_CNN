# -*- coding: utf-8 -*-
from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split


def check_sub_dir():
    # 하위 폴더 경로를 확인한다.
    test_for_categories_list = glob.glob("./my_food/*/*/")
    print(test_for_categories_list)
    return test_for_categories_list


def get_categories():
    # 폴더 경로에서 음식 이름을 추출하여 카테고리 배열로 만든다.
    test_for_categories_list = check_sub_dir()

    categories = []
    for food_dir in test_for_categories_list:
        food = food_dir.split('\\')
        categories.append(food[2])

    print(categories)
    return categories, test_for_categories_list


def make_numpy_data(categories, test_for_categories_list):
    # 데이터세트를 멀티라벨 분류한다.
    caltech_dir = "./my_food"
    nb_classes = len(categories)

    image_w = 64
    image_h = 64

    pixels = image_h * image_w * 3

    X = []
    y = []

    for idx, food_dir in enumerate(test_for_categories_list):

        # one-hot 돌리기.
        label = [0 for i in range(nb_classes)]
        label[idx] = 1

        food_array = food_dir.split('\\')
        food = food_array[0] + '/' + food_array[1] + '/' + food_array[2]

        # image_dir = caltech_dir + "/" + food
        image_dir = food
        files = glob.glob(image_dir + "/*.jpg")
        files.extend(glob.glob(image_dir + '/*.jpeg'))
        files.extend(glob.glob(image_dir + '/*.png'))
        files.extend(glob.glob(image_dir + '/*.gif'))
        print(food, " 파일 길이 : ", len(files))
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)

            X.append(data)
            y.append(label)

            if i % 700 == 0:
                print(food, " : ", f)

    X = np.array(X)
    y = np.array(y)
    # 1 0 0 0 이면 airplanes
    # 0 1 0 0 이면 buddha 이런식

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    xy = (X_train, X_test, y_train, y_test)
    np.save("./numpy_data/multi_image_data.npy", xy)

    print("ok", len(y))


# 메인함수
if __name__ == '__main__':
    categories, test_for_categories_list = get_categories()
    make_numpy_data(categories, test_for_categories_list)