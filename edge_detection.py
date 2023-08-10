def edge_detection(img):

    import cv2
    import numpy as np

    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (11, 11), 0)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)

    canny = cv2.Canny(img, 100, 150)

    cv2.imshow("Image", img)

    cv2.imshow("Canny", canny)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()
