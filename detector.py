from rotate import *


def make_clothes(img, style, file_path, relevant_keypnts):
    # upper left arm
    try:
        img = insert_straight_bone(
            style + file_path[0],
            (relevant_keypnts[0][0], relevant_keypnts[0][1]),
            (relevant_keypnts[1][0], relevant_keypnts[1][1]),
            img
        )
    except IndexError:
        return img
    # upper right arm
    try:
        img = insert_straight_bone(
            style + file_path[2],
            (relevant_keypnts[2][0] - 2, relevant_keypnts[2][1] - 1),
            (relevant_keypnts[3][0], relevant_keypnts[3][1]),
            img
        )
    except IndexError:
        return img
    try:
        img = insert_straight_bone(
            style + file_path[4],
            (relevant_keypnts[4][0], relevant_keypnts[4][1]),
            (relevant_keypnts[5][0], relevant_keypnts[5][1]),
            img
        )
    except IndexError:
        return img
    # left bottom leg
    try:
        img = insert_straight_bone(
            style + file_path[6],
            (relevant_keypnts[6][0], relevant_keypnts[6][1]),
            (relevant_keypnts[7][0], relevant_keypnts[7][1]),
            img
        )
    except IndexError:
        return img
    # right top leg
    try:
        img = insert_straight_bone(
            style + file_path[8],
            (relevant_keypnts[8][0], relevant_keypnts[8][1]),
            (relevant_keypnts[9][0], relevant_keypnts[9][1]),
            img
        )
    except IndexError:
        return img
    # right bottom leg
    try:
        img = insert_straight_bone(
            style + file_path[10],
            (relevant_keypnts[10][0], relevant_keypnts[10][1]),
            (relevant_keypnts[11][0], relevant_keypnts[11][1]),
            img
        )
    except IndexError:
        return img
    try:
        img = insert_rect_bone(
            style + file_path[12],
            (relevant_keypnts[12][0] - 1, relevant_keypnts[12][1] - 13),
            (relevant_keypnts[13][0] + 1, relevant_keypnts[13][1] - 13),
            (relevant_keypnts[14][0], relevant_keypnts[14][1]),
            (relevant_keypnts[15][0], relevant_keypnts[15][1]),
            img
        )
    except IndexError:
        return img
    # final
    return img


def check_landmark(res, img, kit_path):
    file_path = [
        "upperleftarm.png", "0",
        "upperrightarm.png", "1",
        "upperrightleg.png", "2",
        "lowerrightleg.png", "3",
        "upperleftleg.png", "4",
        "lowerleftleg.png", "5",
        "mainbody.png"
    ]
    selected_keypoint_indices = [
        12, 14,  # upper left arm
        11, 13,  # upper right arm
        23, 25,  # upper right leg
        27, 25,  # lower right leg
        24, 26,  # upper left leg
        28, 26,  # lower left leg
        12, 11, 24, 23  # main body
    ]
    height, width = img.shape[:-1]
    if not res.pose_landmarks:
        print('Body not detected!!!')
        return 0
    else:
        pose_landmark = res.pose_landmarks
        # for pose_landmark in results.pose_landmarks:
        values = np.array(pose_landmark.landmark)
        pose_keypnts = np.zeros((len(values), 2))

        for idx, value in enumerate(values):
            pose_keypnts[idx][0] = value.x
            pose_keypnts[idx][1] = value.y

        # Convert normalized points to image coordinates
        pose_keypnts = pose_keypnts * (width, height)
        pose_keypnts = pose_keypnts.astype('int')

        # check the left-right archilles
        if pose_keypnts[31][1] > pose_keypnts[32][1]:
            selected_keypoint_indices = [
                12, 14,  # upper left arm
                11, 13,  # upper right arm
                24, 26,  # upper left leg
                28, 26,  # lower left leg
                23, 25,  # upper right leg
                27, 25,  # lower right leg
                12, 11, 24, 23  # main body
            ]
            file_path = [
                "upperleftarm.png", "0",
                "upperrightarm.png", "1",
                "upperleftleg.png", "4",
                "lowerleftleg.png", "5",
                "upperrightleg.png", "2",
                "lowerrightleg.png", "3",
                "mainbody.png"
            ]

        relevant_keypnts = []

        for i in selected_keypoint_indices:
            if pose_keypnts[i][0] >= width or pose_keypnts[i][0] <= 0:
                continue
            if pose_keypnts[i][1] >= height or pose_keypnts[i][1] <= 0:
                continue
            relevant_keypnts.append(pose_keypnts[i])
        # make clothes
        return make_clothes(img, kit_path, file_path, relevant_keypnts)
