# підключення необхідних бібліотек
import cv2
import numpy as np

sources = {'video1': "data/Video/Hourglass.mp4", 'video2': "data/Video/Hourglass.mp4", "web": 0}


# функція виділення меж canny (стор. 104 - 107)
def canny_edge_detection(image, t_lower=100, t_upper=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray_image, t_lower, t_upper)
    return edge


# функція виділення меж sobel (стор. 104 - 107)
def sobel_edge_detection(image, k_size=3):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, -1, 1, 0, k_size)
    sobel_y = cv2.Sobel(gray_image, -1, 0, 1, k_size)
    sobel_xy = sobel_x + sobel_y
    return sobel_xy


# функція виділення меж prewitt (стор. 104 - 107)
def prewitt_edge_detection(image, axis='x'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if axis == 'x':
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    elif axis == 'y':
        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    else:
        return image.copy()
    return cv2.filter2D(gray_image, -1, kernel)


# функція розмивання (звуження) (стор. 108 - 109)
def erode(image, ksize=3, iterations=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(gray_image, kernel, iterations=iterations)


# функція розтягування (розширення) (стор. 108 - 109)
def dilate(image, ksize=3, iterations=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(gray_image, kernel, iterations=iterations)


# функція порогової бінаризації  (стор. 127 - 129)
def binarization(image, threshold=127):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return thresh


# функція детектування кутів Ши-Томасі  (стор. 140)
def corner_detector(image, max_corners=5, quality_level=0.01, min_dist=20):
    new_image = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_image, max_corners, quality_level, min_dist)
    corners = np.float32(corners)
    for item in corners:
        x, y = item[0]
        cv2.circle(new_image, (int(x), int(y)), 5, 255, -1)
    return new_image


def main():
    cap = cv2.VideoCapture(sources.get('video1'))
    # Перевірка готовності веб-камери
    while cap.isOpened():
        # Запис фреймів
        ret, frame = cap.read()
        # При виникненні помилці запису
        if not ret:
            print("Помилка запису фрейму!")
            break

        # Виконання операції за варіантом
        # frame_changed = canny_edge_detection(frame, t_lower=100, t_upper=200)
        # frame_changed = sobel_edge_detection(frame, k_size=5)
        # frame_changed = prewitt_edge_detection(frame, axis='x')
        # frame_changed = erode(frame, ksize=3, iterations=1)
        # frame_changed = dilate(frame, ksize=5, iterations=3)
        # frame_changed = binarization(frame, threshold=127)
        frame_changed = corner_detector(frame, max_corners=20, quality_level=0.01, min_dist=50)

        # Відображення результату
        cv2.imshow('frame', frame)
        cv2.imshow('frame_changed', frame_changed)
        if cv2.waitKey(25) == ord('q'):
            break
    # Завершуємо запис у кінці роботи
    cap.release()
    cv2.destroyAllWindows()


# при запуску як головного файлу
if __name__ == '__main__':
    main()
