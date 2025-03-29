
import numpy as np
import cv2




def get_image(width, height,raw_image):
    image = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
    return image


def pixel_difference(pixel1, pixel2):
    squared_dist = np.sum((pixel1 - pixel2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def colour_threshold(image, colour):
    binary_image = np.zeros(image.shape, dtype=np.uint8)
    binary_image[:][:][4] = 255  # alpha channel
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            pixel = image[r][c][:-1]
            difference = pixel_difference(colour, pixel)
            if difference < 50:  # 30 from c code
                binary_image[r][c][:-1] = 255
    return binary_image


def colour_threshold_cv2(image, colour, error):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = colour - error / 2
    higher = colour + error / 2
    mask = cv2.inRange(hsv, lower, higher)

    return mask


def calculate_normalized_average_col(binary_image):
    th_idx = np.where(binary_image == 255)

    if not th_idx:
        print("WARNING: Binary image ALL zeros")
        return None, None  # TODO: improve
    else:
        average_column = np.average(th_idx[1])
        normalized_column = (
            average_column - binary_image.shape[1] / 2
        ) / binary_image.shape[1]
        # Ranges from [-0.5, 0.5]
        return normalized_column, int(average_column)


def display_binary_image( binary_image, average_column):
    # Image to display
    binary_image_gbra = np.dstack(
        (
            binary_image,
            binary_image,
            binary_image,
        )
    )
    # Highlight average column
    binary_image_gbra[:, average_column, 0] = 255  # red
    binary_image_gbra[:, average_column, 1] = 0  # green
    binary_image_gbra[:, average_column, 2] = 0  # blue

    cv2.imshow("image", binary_image_gbra)


if __name__ == "__main__":
    image = cv2.imread("/home/djoker/Carla/Bin/dataset/session_20250322_110901/images/frame_20250322_110937_324.jpg")

    # yellow_pixel = np.array([95, 187, 203])  # road yellow (BGR format)
    yellow_pixel = np.array([25, 127, 127])
    error = np.array([5, 255, 255])

    binary_image = colour_threshold_cv2(image, yellow_pixel, error)
    normalized_column, average_column = calculate_normalized_average_col(binary_image)
            

    display_binary_image(binary_image, average_column)  

    cv2.imshow("binary image", binary_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
    