import streamlit as st
from roboflow import Roboflow
import cv2
from PIL import Image

# create Roboflow object
rf = Roboflow(api_key="CnSgXUPmgAxiaKeC1tEf")
project = rf.workspace().project("yolodetect-m4gkz")
model = project.version(2).model

st.header("Идентификация пиломатериалов")
# Загрузка изображений
uploaded_files = st.file_uploader("Выберите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) == 2:
    # Преобразование изображений в объекты PIL и сохранение их в виде файлов
    filenames = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        filename = uploaded_file.name
        filenames.append(filename)
        st.image(image, caption=filename, use_column_width=True)
        image.save(f"{filename}")

    # Обработка изображений и выравнивание
    model_confidence = 40
    model_overlap = 30
    model_class = 'board'
    board_images = []
    for filename in filenames:
        # Получение результата предсказания для класса "board"
        result = [p for p in model.predict(filename, confidence=model_confidence, overlap=model_overlap).json()['predictions'] if p['class'] == model_class][0]
        # Извлечение координат рамки и корректировка смещения
        x, y, w, h = result['x'] - 400, result['y'] - 150, result['width'], result['height']
        # Обрезка изображения по рамке
        im = Image.open(result['image_path'])
        board_im = im.crop((x, y, x + w, y + h))
        board_im.save(f"board_{filename}")

        # Выравнивание изображений
        board_image = cv2.imread(f"board_{filename}")
        board_images.append(board_image)

    if board_images[0].shape != board_images[1].shape:
        board_images[1] = cv2.resize(board_images[1], (board_images[0].shape[1], board_images[0].shape[0]))

    # Отображение выровненных изображений
    st.header("Отображение выровненных изображений")
    st.image(board_images, caption=[f"board_{filename}" for filename in filenames], use_column_width=True)
   
    # Детекция границ на изображениях с помощью алгоритма Canny
    edges1 = cv2.Canny(cv2.cvtColor(board_images[0], cv2.COLOR_BGR2GRAY), threshold1=100, threshold2=200)
    edges2 = cv2.Canny(cv2.cvtColor(board_images[1], cv2.COLOR_BGR2GRAY), threshold1=100, threshold2=200)

    # Создание детектора контуров
    if cv2.__version__.startswith('3.'):
            contour_detector = cv2.xfeatures2d.SIFT_create()
    else:
             contour_detector = cv2.SIFT_create()


    # Нахождение ключевых точек и дескрипторов для контуров
    keypoints1, descriptors1 = contour_detector.detectAndCompute(edges1, None)
    keypoints2, descriptors2 = contour_detector.detectAndCompute(edges2, None)

    # Сравнение дескрипторов изображений
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Отображение ключевых точек на изображениях
    img1_kp = cv2.drawKeypoints(board_images[0], keypoints1, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(board_images[1], keypoints2, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Отображение изображений с ключевыми точками
    st.header("Изображения с ключевыми точками")
    st.image(img1_kp, use_column_width=True)
    st.image(img2_kp, use_column_width=True)

    # Применение порогового значения для отбора наилучших соответствий
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Создание объекта Matcher
    matcher = cv2.BFMatcher()

# Нахождение соответствий между ключевыми точками
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# Отбор лучших соответствий
    good_matches = []
    for m, n in matches:
              if m.distance < 0.75 * n.distance:
                  good_matches.append(m)

# Отображение соответствий на изображениях
    img_matches = cv2.drawMatches(board_images[0], keypoints1, board_images[1], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Отображение изображений с соответствиями
    st.header("Изображения с соответствиями")
    st.image(img_matches, use_column_width=True)

    # Вывод сходства или расхождения двух изображений торцов доски
    st.header("Вывод сходства")
    if len(good_matches) >= 20:
        st.header("ЭТО ОДНА ДОСКА")
    else:
        st.header("ЭТО РАЗНЫЕ ДОСКИ")