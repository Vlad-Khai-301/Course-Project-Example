# підключення необхідних бібліотек
import cv2
import numpy as np

sources = {'video1': "data/Video/Hourglass.mp4", 'video2': "data/Video/Hourglass.mp4", "web": 0}


# Низькочастотна фільтрація
def LF_filtration(image, kernel_size=3, center_coef=1):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size + center_coef - 1)
    center_index = int((kernel_size - 1) * 0.5)
    kernel[center_index][center_index] *= center_coef
    return cv2.filter2D(image, -1, kernel)


# Високочастотна фільтрація
def HF_filtration(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) * (-1)
    center_index = int((kernel_size - 1) * 0.5)
    kernel[center_index][center_index] = kernel_size * kernel_size
    return cv2.filter2D(image, -1, kernel)


# Фільтрація з ефектом зсуву
def motion_blur(image, kernel_size=3):
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    return cv2.filter2D(image, -1, kernel_motion_blur)


# фільтрація з підкресленням меж
def sharpening_filtration(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    center_index = int((kernel_size - 1) * 0.5)
    kernel[center_index][center_index] = (kernel_size * kernel_size - 2) * -1
    return cv2.filter2D(image, -1, kernel)


# Фільтрація з перетворенням на рельєфне зображення
def embossing_filtration(image, kernel_size=3):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Південно-західний регіон
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    for index_r, values in enumerate(kernel):
        for index_c, value in enumerate(values):
            if index_c > index_r:
                kernel[index_r][index_c] = -1
            elif index_r > index_c:
                kernel[index_r][index_c] = 1
    print(kernel)
    return cv2.filter2D(gray_img, -1, kernel) + 128


# Фільтр Гауса
def gaussian_filtration(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Білатеральний фільтр
def bilateral_filtration(image, filter_size=9, sigmaValues=75):
    return cv2.bilateralFilter(image, filter_size, sigmaValues, sigmaValues)


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
        # frame_changed = LF_filtration(frame, kernel_size=9, center_coef=1)
        # frame_changed = HF_filtration(frame, kernel_size=7)
        # frame_changed = motion_blur(frame, kernel_size=7)
        # frame_changed = sharpening_filtration(frame, kernel_size=5)
        frame_changed = embossing_filtration(frame, kernel_size=5)
        # frame_changed = gaussian_filtration(frame, kernel_size=11)
        # frame_changed = bilateral_filtration(frame, filter_size=9, sigmaValues=75)

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
