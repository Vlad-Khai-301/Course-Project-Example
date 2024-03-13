# підключення необхідних бібліотек
import cv2
import numpy as np

sources = {'video1': "data/Video/Hourglass.mp4", 'video2': "data/Video/Hourglass.mp4", "web": 0}


# функція повороту зображення (стор. 132 - 133)
def rotate_image(image, angle=0):
    num_rows, num_cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    img_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    return img_rotation


# функція переносу зображення (стор. 134)
def parallel_transfer(image, left=0, top=0):
    num_rows, num_cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, left], [0, 1, top]])
    img_translation = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    return img_translation


# функція скісу зображення (стор. 134 - 135)
def bevel(image, coef_1=1.0, coef_2=0.0):
    rows, cols = image.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    dst_points = np.float32([[0, 0], [int(coef_1 * (cols - 1)), 0],
                             [int(coef_2 * (cols - 1)), rows - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(image, affine_matrix, (cols, rows))
    return img_output


# функція дзеркального відображення зображення (стор. 135)
def mirror(image):
    rows, cols = image.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(image, affine_matrix, (cols, rows))
    return img_output


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

        # Геометричні перетворення зображення (фрейму)
        # frame_rotate = rotate_image(frame, 30)
        # frame_transfer = parallel_transfer(frame, 100)
        # frame_bevel = bevel(frame, 0.6, 0.4)
        frame_mirror = mirror(frame)

        # Відображення результату
        cv2.imshow('frame', frame)
        # cv2.imshow('frame_rotated', frame_rotate)
        # cv2.imshow('frame_transfered', frame_transfer)
        # cv2.imshow('frame_bevel', frame_bevel)
        cv2.imshow('frame_mirrored', frame_mirror)
        if cv2.waitKey(25) == ord('q'):
            break
    # Завершуємо запис у кінці роботи
    cap.release()
    cv2.destroyAllWindows()


# при запуску як головного файлу
if __name__ == '__main__':
    main()
