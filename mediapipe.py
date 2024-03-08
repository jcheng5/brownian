import math
from functools import reduce
from operator import add
from statistics import mean
import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def hand_to_camera_eye(hands, detect_ok=False):
    # TODO: Pointing straight at the camera should not change (much) from the default
    # position
    # TODO: Use spherical coordinates instead of cartesian

    # x: left (0) to right (1)
    # y: top (0) to bottom (1)
    # z: close (more negative) to far (more positive)

    width = hands["image"]["width"]
    height = hands["image"]["height"]

    # Note that the handedness is flipped (0 is normally left), because this app
    # is designed to be used with a selfie cam, so the image is mirrored
    left_hand = hands["multiHandedness"][0]["index"] != 0
    hand = hands["multiHandLandmarks"][0]

    def hand_coords(landmark: int):
        lm = hand[landmark]
        # normalize y scale to match x scale (which z already does)
        return np.array([lm["x"], lm["y"] * height / width, lm["z"]])

    def rel_hand(start_pos: int, end_pos: int):
        return np.subtract(hand_coords(start_pos), hand_coords(end_pos))

    if detect_ok:
        # If the distance between the thumbtip and index finger tip are pretty close,
        # ignore the hand. (Using "OK" sign to pause hand tracking)
        ok_dist = np.linalg.norm(rel_hand(THUMB_TIP, INDEX_FINGER_TIP))
        ref_dist = np.linalg.norm(rel_hand(INDEX_FINGER_TIP, INDEX_FINGER_DIP))
        if ok_dist < ref_dist * 2:
            return None

    if left_hand:
        p1 = rel_hand(PINKY_MCP, WRIST)
        p2 = rel_hand(INDEX_FINGER_MCP, WRIST)
    else:
        p1 = rel_hand(INDEX_FINGER_MCP, WRIST)
        p2 = rel_hand(PINKY_MCP, WRIST)
    up = rel_hand(WRIST, MIDDLE_FINGER_MCP)
    normal_vec = np.cross(p1, p2)
    normal_unit_vec = normal_vec / np.linalg.norm(normal_vec)

    # Invert to convert from the direction we're looking towards,
    # to the direction the camera is located
    eye_vec = normal_unit_vec * -1.0

    # Zoom out (get further from the origin)
    eye_vec *= 2

    # Less precision
    eye_vec = eye_vec.round(3)

    return {
        "eye": {
            # Rotate axes to match plotly
            "x": eye_vec[2],
            "y": eye_vec[0],
            "z": eye_vec[1],
        },
        "up": {
            "x": up[2],
            "y": -up[0],
            "z": up[1],
        },
    }


def info_smoother(points):
    return dict(
        eye=dict(
            x=mean([p["eye"]["x"] for p in points]),
            y=mean([p["eye"]["y"] for p in points]),
            z=mean([p["eye"]["z"] for p in points]),
        ),
        up=dict(
            x=mean([p["up"]["x"] for p in points]),
            y=mean([p["up"]["y"] for p in points]),
            z=mean([p["up"]["z"] for p in points]),
        ),
    )
