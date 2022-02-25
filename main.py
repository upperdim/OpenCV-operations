from matplotlib import pyplot as plt
import numpy as np
import cv2


def basics():
    # Read image
    img_url = 'starry_night.jpg'
    param = -1  # optional parameter (1 = colored, 0 = grayscale, -1 = unchanged)
    # cv2.IMREAD_COLOR or 1        - Alpha layer will be removed
    # cv2.IMREAD_GRAYSCALE or 0    - Alpha layer will remain
    # cv2.IMREAD_UNCHANGED or -1   - Alpha layer will be removed, obviously
    img = cv2.imread(img_url, param)

    # Show image
    cv2.imshow('Sample Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Access image properties
    print(img.shape)  # (rows, cols, channels)

    # Convert between color spaces
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    gray_img = cv2.resize(gray_img, (800, 600))

    # Put text onto the image
    text_coords = (50, 200)
    font_scale = 4
    text_color = (255, 127, 0)
    text_thickness = 4
    gray_img = cv2.putText(gray_img, 'Sample Text', text_coords, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

    # Draw line onto the image
    start_point = (300, 300)
    end_point = (450, 450)
    line_color = (128, 255, 0)
    line_thickness = 10
    gray_img = cv2.line(gray_img, start_point, end_point, line_color, line_thickness)

    # Draw circle onto the image
    circle_center = (300, 40)
    circle_radius = 120
    circle_color = (0, 128, 255)
    circle_thickness = 6
    gray_img = cv2.circle(gray_img, circle_center, circle_radius, circle_color, circle_thickness)

    # Draw rectangle onto the image
    rect_start_coord = (20, 20)
    rect_end_coord = (780, 580)
    rect_color = (255, 255, 255)
    rect_thickness = 20
    gray_img = cv2.rectangle(gray_img, rect_start_coord, rect_end_coord, rect_color, rect_thickness)

    # Draw ellipse onto the image
    el_color = (255, 0, 255)
    el_thickness = 5
    el_center_coords = (120, 100)
    el_axes_length = (100, 50)
    el_angle = 30
    el_start_angle = 0
    el_end_angle = 360
    gray_img = cv2.ellipse(gray_img, el_center_coords, el_axes_length, el_angle, el_start_angle, el_end_angle, el_color, el_thickness)

    # Save image
    out_dir = '.\\out_imgs\\'
    out_img_name = 'gray_img.jpg'
    cv2.imwrite(out_img_name, gray_img)


def webcam():
    cap = cv2.VideoCapture(0)
    while (True):
        ret_, frame = cap.read()
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()


def webcam_cap():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Camera could not open.')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_codec = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = 30
    video_output = cv2.VideoWriter('captured_video.mp4', video_codec, video_fps, (frame_width, frame_height))
    while (True):
        ret, frame = cap.read()
        if ret is True:
            video_output.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    print('The video was saved successfully.')


def play_vid():
    vid_file = 'captured_video.mp4'
    cap = cv2.VideoCapture(vid_file)
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def img_ops():
    img = cv2.imread('starry_night.jpg')

    # Access to a pixel in the image
    px = img[100, 100]
    px_blue_val = img[100, 100, 0]
    print(f'BGR = {px}, Blue value = {px_blue_val}')

    # Replace pixels
    img[100, 100] = [255, 127, 0]

    # Image shape
    img_file = 'wikimedia_alpha.png'
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    alpha_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    print(f'RGB shape        = {img.shape}')
    print(f'ARGB shape       = {alpha_img.shape}')
    print(f'Grayscale shape  = {gray_img.shape}')

    # Data type
    print(f'Image data type = {img.dtype}')

    # Size
    print(f'Image size (rows * cols * channels) = {img.size}')

    # Setting region of image (ROI)
    roi = cv2.selectROI(img)
    print(roi)  # prints (start_x, start_y, width_of_roi, height_of_roi)
    roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    cv2.imshow('ROI Image', roi_cropped)
    cv2.imwrite('cropped.jpg', roi_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Split and merge images
    img_g, img_b, img_r = cv2.split(img)
    cv2.imshow('Green Part of the Image', img_g)
    cv2.imshow('Blue Part of the Image',  img_b)
    cv2.imshow('Red Part of the Image',   img_r)
    cv2.waitKey(0)
    img_merge = cv2.merge((img_g, img_b, img_r))
    cv2.imshow('Image after merger of 3 colors', img_merge)
    cv2.waitKey(0)


def change_color_scheme():
    img_file = 'starry_night.jpg'
    img = cv2.imread(img_file)

    color_change = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cv2.imshow('Changed color scheme', color_change)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def merge_images():
    src1 = cv2.imread('starry_night.jpg', cv2.IMREAD_COLOR)
    src2 = cv2.imread('wikimedia_alpha.png', cv2.IMREAD_COLOR)
    img1 = cv2.resize(src1, (800, 600))
    img2 = cv2.resize(src2, (800, 600))

    blended_img = cv2.addWeighted(img1, 1, img2, 0.5, 0.25)

    cv2.imshow('Blended/additive image', blended_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filters():
    img = cv2.imread('starry_night.jpg')
    k_sharped = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, k_sharped)
    cv2.imshow('Filtered Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def threshold():
    img = cv2.imread('starry_night.jpg')
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    canny_img = cv2.Canny(img, 50, 100)

    cv2.imshow('Treshold Image', thresh)
    cv2.imshow('Canny Image', canny_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_shape_detection():
    img = cv2.imread('shapes.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting threshold of the grayscale image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Contours using findContours()
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        # Skip first contour
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (255, 0, 255), 5)

        # Finding the center of different shapes
        x, y = 0, 0
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int( M['m10'] / M['m00'] )
            y = int( M['m01'] / M['m00'] )

        tag_color = (127, 127, 127)
        # Put names of the shapes inside the corresponding shapes
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)
        elif len(approx) == 4:
            cv2.putText(img, 'Quadrilaterla', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)
        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)
        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)
        else:
            cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2)

    cv2.imshow('Shapes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_detection():
    #img = cv2.imread('starry_night.jpg')
    img = cv2.imread('shapes.jpg')
    # HSV (Hue, Saturation, Value), commonly used in color and paint software
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 50, 50])
    lower_green = np.array([40, 40, 40])
    lower_yellow = np.array([10, 100, 20])

    upper_blue = np.array([140, 255, 255])
    upper_green = np.array([70, 255, 255])
    upper_yellow = np.array([25, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
    res_green = cv2.bitwise_and(img, img, mask=mask_green)
    res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

    cv2.imshow('Blue Filtered', res_blue)
    cv2.imshow('Green Filtered', res_green)
    cv2.imshow('Yellow Filtered', res_yellow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def obj_replace():
    img = cv2.imread('starry_night.jpg')
    img_cpy = img.copy()
    mask = np.zeros((100, 300, 3))
    print(mask.shape)

    pos = (200, 200)
    var = img_cpy[200:(200 + mask.shape[0]), 200:(200 + mask.shape[1])] = mask
    cv2.imshow('colring', img_cpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_diff(orig_img, new_img, new_label):
    """Utility function to display original and altered version of the image with user specified label"""
    cv2.imshow('Original', orig_img)
    cv2.imshow(new_label, new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morph_erosion():
    """Result is 1 only if all pixels under kernel is 1"""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    display_diff(img, erosion, 'Erosion')


def morph_dilation():
    """Result is 1 even if only 1 pixel under kernel is 1"""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    display_diff(img, dilation, 'Dilation')


def morph_opening():
    """Opening is actually erosion followed by dilation. Useful for removing noise."""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    display_diff(img, opening, 'Opening')


def morph_closing():
    """Closing is actually dilation followed by erosion. Useful for closing little holes in the object."""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    display_diff(img, closing, 'Closing')


def morph_gradient():
    """Difference between erosion and dilation of the object. Looks like an outline."""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    display_diff(img, gradient, 'Gradient')


def morph_top_hat():
    """Difference between image and opening of the image."""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    display_diff(img, tophat, 'Top Hat')


def morph_black_hat():
    """Difference between image and closing of the image."""
    img = cv2.imread('j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((9, 9), np.uint8)
    black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    display_diff(img, black_hat, 'Black Hat')


def morph_structuring_elems():
    rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    print(f'Rectangular Kernel:\n{rectangular_kernel}')
    print(f'Elliptical Kernel:\n{elliptical_kernel}')
    print(f'Cross Kernel:\n{cross_kernel}')


def main():
    # Uncomment the section which you are interested down below:
    #
    # basics()
    # webcam()
    # webcam_cap()
    # play_vid()
    # img_ops()
    # change_color_scheme()
    # merge_images()
    # filters()
    # threshold()
    # contour_shape_detection()
    # color_detection()
    # obj_replace()
    # morph_erosion()
    # morph_dilation()
    # morph_opening()
    # morph_closing()
    # morph_gradient()
    # morph_top_hat()
    # morph_black_hat()
    # morph_structuring_elems()
    pass


if __name__ == '__main__':
    main()
