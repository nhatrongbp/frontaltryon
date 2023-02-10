# This is a sample Python script.
import cv2
import numpy as np
import numba as nb
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


@nb.njit('(uint8[:,:,::1], uint8[:,:,::1])', parallel=True)
def compute(img, canvas):
    for i in nb.prange(img.shape[0]):
        for j in range(img.shape[1]):
            ir = np.float32(img[i, j, 0])
            ig = np.float32(img[i, j, 1])
            ib = np.float32(img[i, j, 2])
            cr = np.float32(canvas[i, j, 0])
            cg = np.float32(canvas[i, j, 1])
            cb = np.float32(canvas[i, j, 2])
            alpha = np.float32((ir + ig + ib) > 0)
            inv_alpha = np.float32(1.0) - alpha
            cr = inv_alpha * cr + alpha * ir
            cg = inv_alpha * cg + alpha * ig
            cb = inv_alpha * cb + alpha * ib
            canvas[i, j, 0] = np.uint8(cr)
            canvas[i, j, 1] = np.uint8(cg)
            canvas[i, j, 2] = np.uint8(cb)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    img = cv2.imread("tempkit/upperrightarm.png", cv2.IMREAD_UNCHANGED)
    # h, w, _ = img.shape
    w, h, _ = img.shape
    print(w, h)
    # img = cv2.resize(img, (w, int(h/4)))
    img = cv2.resize(img, (w, int(h/5)))
    canvas = np.zeros((w, w, 3), dtype=np.uint8)
    rotation_angle = 0
    width = img.shape[1]
    height = img.shape[0]
    pivot_point = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(pivot_point, -rotation_angle, 1.)

    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]

    rotation_mat[0, 2] += canvas_width / 2 - pivot_point[0]
    rotation_mat[1, 2] += canvas_height / 2 - pivot_point[1]

    rotated_image = cv2.warpAffine(img,
                                   rotation_mat,
                                   (canvas_width, canvas_height))
    compute(rotated_image, canvas)
    cv2.imwrite("awaykit/upperrightarm.png", canvas)
    # cv2.imshow("b", canvas)
    # cv2.waitKey()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("run test.py script, dont run this script!")
    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
