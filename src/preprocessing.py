import cv2
import numpy as np


##load image as grayscale.
def load_grayscale_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image

#normalize pixel from 0-255 to 0-1
def normalize_image(image):
    return image.astype(np.float32) / 255.0

#resize
def resize_image(image, size=(32, 32)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


##crop the image around the visible digit -> basically assume the digit is brigther than the background
def crop_to_digit(image, threshold=0.1, padding=2):
    mask = image > threshold

    if not np.any(mask):
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, image.shape[0] - 1)
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1] - 1)

    cropped = image[y_min:y_max + 1, x_min:x_max + 1]

    return cropped


##padding; if there is a rectangualr image -> make it square
def pad_to_square(image):
    h, w = image.shape

    if h == w:
        return image

    size = max(h, w)
    padded = np.zeros((size, size), dtype=image.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image

    return padded


##The full pipeline looks smt like
    ## 1.normalize image
    ## 2.crop around digit
    ## 3.pad to square
    ## 4.resize
def preprocess_image(
    image,
    target_size=(32, 32),
    use_crop=True,
    use_normalize=True
):

    if use_normalize:
        image = normalize_image(image)

    if use_crop:
        image = crop_to_digit(image)
        image = pad_to_square(image)

    image = resize_image(image, target_size)

    return image.astype(np.float32)