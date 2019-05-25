import os
import string
import math
import cv2
import numpy as np

# import shutil
import requests
import tempfile

from datetime import datetime
from time import sleep, time
from keras.models import load_model

auth_url = 'https://nportal.ntut.edu.tw/authImage.do'
# image_path = 'keras_ntut_verification_code/photo/'
image_path = 'photo/'
# split_width = 40
image_size = (120, 40)
split_width = 40
# model = load_model('keras_ntut_verification_code/nportal_splitted.h5')
model = load_model('nportal_splitted.h5')


def split_letter(im):
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = []
    for contour in contours:
        x += [p[0][0] for p in contour]
    x_max = max(x)
    x_min = min(x)
    width = x_max - x_min
    # print(x_max, x_min)

    offset = (split_width - (x_max - x_min)) // 2
    if x_min < offset:
        start = 0
        end = split_width
    elif x_max + offset >= image_size[0]-1:
        start = image_size[0] - split_width
        end = image_size[0]
    else:
        start = x_min - offset
        end = x_min - offset + split_width

    # print(start, end)

    s_im = im[:, start:end]
    s_im = s_im.reshape(s_im.shape[0], s_im.shape[1], 1)
    return s_im, x_min, width


def ocr_auth(image, debug=False, sampling=False):
    im = cv2.imread(image)
    # im = im.astype('float64')
    # im = cv2.resize(im, image_size)
    u, count = np.unique(
        im.reshape(-1, im.shape[2]), return_counts=True, axis=0)
    colors = list(zip(u, count))
    colors = sorted(colors, key=lambda x: x[1], reverse=True)

    # background_color = s_owo[0][0]
    color_index = 1
    while True:
        a = colors[color_index][0]
        a_im = cv2.inRange(im, a, a)
        a_splitted, a_start, a_width = split_letter(a_im)
        color_index += 1
        if a_width < 40:
            break

    while True:
        b = colors[color_index][0]
        b_im = cv2.inRange(im, b, b)
        b_splitted, b_start, b_width = split_letter(b_im)
        color_index += 1
        if b_width < 40:
            break

    while True:
        c = colors[color_index][0]
        c_im = cv2.inRange(im, c, c)
        c_splitted, c_start, c_width = split_letter(c_im)
        color_index += 1
        if c_width < 40:
            break

    while True:
        d = colors[color_index][0]
        d_im = cv2.inRange(im, d, d)
        d_splitted, d_start, d_width = split_letter(d_im)
        color_index += 1
        if d_width < 40:
            break

    # b = colors[2][0]
    # b_im = cv2.inRange(im, b, b)
    # b_splitted, b_start, _ = split_letter(b_im)

    # c_im = cv2.inRange(im, c, c)
    # c = colors[3][0]
    # c_splitted, c_start, _ = split_letter(c_im)

    # d = colors[4][0]
    # d_im = cv2.inRange(im, d, d)
    # d_splitted, d_start, _ = split_letter(d_im)

    # a_splitted // 255
    # b_splitted // 255
    # c_splitted // 255
    # d_splitted // 255

    a_predict = model.predict(a_splitted.reshape(
        1, a_splitted.shape[0], a_splitted.shape[1], a_splitted.shape[2]), batch_size=1)
    b_predict = model.predict(b_splitted.reshape(
        1, b_splitted.shape[0], b_splitted.shape[1], b_splitted.shape[2]), batch_size=1)
    c_predict = model.predict(c_splitted.reshape(
        1, c_splitted.shape[0], c_splitted.shape[1], c_splitted.shape[2]), batch_size=1)
    d_predict = model.predict(d_splitted.reshape(
        1, d_splitted.shape[0], d_splitted.shape[1], d_splitted.shape[2]), batch_size=1)

    if debug:
        cv2.imwrite('a.png', a_splitted)
        cv2.imwrite('b.png', b_splitted)
        cv2.imwrite('c.png', c_splitted)
        cv2.imwrite('d.png', d_splitted)
    # print(np.argmax(a_predict[0]))
    result = list(zip([a_start, b_start, c_start, d_start], [string.ascii_uppercase[np.argmax(a_predict)], string.ascii_uppercase[np.argmax(
        b_predict)], string.ascii_uppercase[np.argmax(c_predict)], string.ascii_uppercase[np.argmax(d_predict)]], [a_splitted, b_splitted, c_splitted, d_splitted]))
    # print(result)
    result = sorted(result, key=lambda x: x[0])
    auth = [x[1] for x in result]
    # print(result)
    auth = ''.join(auth)
    print(auth)

    # full_image = a_im | b_im | c_im | d_im
    full_image = im

    if sampling:
        for i in result:
            # date = datetime.now()
            # date_string = date.strftime(r'%Y%m%d%H%M%S')
            date_string = str(time())
            cv2.imwrite('testing/'+i[1]+'/'+date_string+'.png', i[2])
            # cv2.imwrite('a.png', a_splitted)
            # cv2.imwrite('b.png', b_splitted)
            # cv2.imwrite('c.png', c_splitted)
            # cv2.imwrite('d.png', d_splitted)

    if debug:
        # print(result)
        cv2.imwrite('test.png', full_image)
    else:
        date = datetime.now()
        date_string = date.strftime(r'%Y%m%d%H%M%S')
        # cv2.imwrite('labeled/'+result+'-'+date_string+'.png', full_image)

        cv2.imwrite('testing/Full/'+auth+'-'+date_string+'.png', full_image)


if __name__ == "__main__":
    # dir_list = os.listdir(image_path)
    # # print(dir_list)
    # images = [f for f in dir_list if f[-4:] == '.png']
    # # print(images)

    # for image in images:
    #     ocr_auth(image_path+image)
    # ocr_auth('temp.png', debug=True)

    for i in range(10):
        print('image:', i)
        r = requests.get(auth_url, stream=True)
        if r.status_code == 200:
            with open('temp.png', 'wb') as f:
                f.write(r.content)
                f.close()
            ocr_auth('temp.png')
            sleep(0.3)
