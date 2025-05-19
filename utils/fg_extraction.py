import cv2 as cv

def fg_extraction(image_name, mask_name):
    img = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img, 5)
    #
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv.THRESH_BINARY, 11, 2)
    #
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # images = [img, th1, th2, th3]
    # print(np.unique(th1))
    cv.imwrite(mask_name, th1)