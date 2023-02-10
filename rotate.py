import cv2
import numpy as np
import numba as nb


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


def euclid_distance(a, b):
    point1 = np.array((a[0], a[1], 0))
    point2 = np.array((b[0], b[1], 0))
    # calculating Euclidean distance
    # using linalg.norm()
    dist = np.linalg.norm(point1 - point2)
    return dist


def insert_straight_bone(img_path, a, b, new_img):
    d = int(euclid_distance(a, b) + 1)
    d += 5
    # print("distance: ", d)
    canvas = np.zeros((d, d, 3), dtype=np.uint8)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (d, d))

    # doc truoc ngang sau:
    # rotation_angle = int(np.degrees(np.arctan2(b[0]-a[0], b[1]-a[1])))
    # ngang truoc doc sau:
    rotation_angle = int(np.degrees(np.arctan2(a[1]-b[1], a[0]-b[0])))
    # print("angle:", rotation_angle)
    width = image.shape[1]
    height = image.shape[0]
    pivot_point = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(pivot_point, -rotation_angle, 1.)

    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]

    rotation_mat[0, 2] += canvas_width / 2 - pivot_point[0]
    rotation_mat[1, 2] += canvas_height / 2 - pivot_point[1]

    rotated_image = cv2.warpAffine(image,
                                   rotation_mat,
                                   (canvas_width, canvas_height))
    compute(rotated_image, canvas)
    # canvas = cv2.resize(canvas, (500, 500))
    # new_img = cv2.imread(dst_path)
    mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
    # print("middle: ", mid)
    mid[0] = int(mid[0] - d / 2)
    mid[1] = int(mid[1] - d / 2)
    # print("upper left: ", int(mid[0]), int(mid[1]))
    # print("lower right: ", int(mid[0]) + d, int(mid[1]) + d)
    try:
        # new_img[0:d, 0:d][np.where((canvas != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # new_img[0:d, 0:d] += canvas
        new_img[mid[1]:mid[1] + d, mid[0]:mid[0] + d][np.where((canvas != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        new_img[mid[1]:mid[1] + d, mid[0]:mid[0] + d] += canvas
        # cv2.imshow("c", new_img)
        # cv2.waitKey()
        return new_img
    except IndexError:
        # cv2.imshow("c", new_img)
        # cv2.waitKey()
        return new_img


def insert_rect_bone(img_path, a, b, c, d, new_img):
    w = int(euclid_distance(a, b) + 1)
    w += 5
    upper_middle = ((a[0]+b[0])/2, (a[1]+b[1])/2)
    lower_middle = ((c[0]+d[0])/2, (c[1]+d[1])/2)
    h = int(euclid_distance(upper_middle, lower_middle) + 1)
    h += 5
    # print("distance: ", d)
    canvas = np.zeros((h, h, 3), dtype=np.uint8)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (h, w))

    # doc truoc ngang sau:
    # rotation_angle = int(np.degrees(np.arctan2(b[0]-a[0], b[1]-a[1])))
    # ngang truoc doc sau:
    rotation_angle = int(np.degrees(np.arctan2(upper_middle[1]-lower_middle[1], upper_middle[0]-lower_middle[0])))
    # print("angle:", rotation_angle)
    width = image.shape[1]
    height = image.shape[0]
    pivot_point = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(pivot_point, -rotation_angle, 1.)

    canvas_height = canvas.shape[0]
    canvas_width = canvas.shape[1]

    rotation_mat[0, 2] += canvas_width / 2 - pivot_point[0]
    rotation_mat[1, 2] += canvas_height / 2 - pivot_point[1]

    rotated_image = cv2.warpAffine(image,
                                   rotation_mat,
                                   (canvas_width, canvas_height))
    compute(rotated_image, canvas)
    # canvas = cv2.resize(canvas, (500, 500))
    # new_img = cv2.imread(dst_path)
    mid = [(upper_middle[0] + lower_middle[0]) / 2, (upper_middle[1] + lower_middle[1]) / 2]
    # print("middle: ", mid)
    mid[0] = int(mid[0] - h / 2)
    mid[1] = int(mid[1] - h / 2)
    # print("upper left: ", int(mid[0]), int(mid[1]))
    # print("lower right: ", int(mid[0]) + d, int(mid[1]) + d)
    try:
        # new_img[0:d, 0:d][np.where((canvas != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # new_img[0:d, 0:d] += canvas
        new_img[mid[1]:mid[1] + h, mid[0]:mid[0] + h][np.where((canvas != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        new_img[mid[1]:mid[1] + h, mid[0]:mid[0] + h] += canvas
        # cv2.imshow("c", new_img)
        # cv2.waitKey()
        return new_img
    except IndexError:
        # cv2.imshow("c", new_img)
        # cv2.waitKey()
        return new_img


# cv2.imshow("a", insert_straight_bone("Myproject.png", (200, 100), (100, 200), cv2.imread("03615_00.jpg")))
# # doc truoc ngang sau
# cv2.waitKey()


def angel(a, b):
    print(np.degrees(np.arctan2(b[0]-a[0], b[1]-a[1])))


# angel((200, 100), (100, 200))
# angel((100, 100), (200, 100))
# angel((200, 200), (100, 100))
