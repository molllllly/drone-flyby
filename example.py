%%writefile example.py

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from dtos import (
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)
from utils import decode_view


VERSION = "affine_tracks_v2"

logger = logging.getLogger(__name__)
logger.warning("EXAMPLE VERSION: %s", VERSION)


# ============================================================
# MODEL
# ============================================================

MODEL_WEIGHTS = "runs/detect/train-4/weights/best.pt"

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()

# Proposal threshold, NOT final output threshold.
PROPOSAL_CONF = 0.01

NMS_IOU = 0.50
MAX_DETECTIONS = 250

IMAGE_W = 960
IMAGE_H = 540


# ============================================================
# FAST MOTION ESTIMATION
# ============================================================

MAX_CORNERS = 700
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 8

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3

MIN_FLOW_POINTS = 20

MIN_AFFINE_INLIER = 0.45
RANSAC_THRESHOLD = 2.5


# ============================================================
# TEMPORAL CONFIRMATION
# ============================================================

RECENT_WINDOW = 4

# Normal low-confidence track:
# at least 3 observations in recent 4 frames.
MIN_RECENT_HITS = 3

# Allows ~3 observations around conf ~= 0.01.
MIN_RECENT_CONF_SUM = 0.025

# Geometry must also agree.
MIN_CONFIRM_MATCH_IOU = 0.15


# Strong detections can enter faster.
STRONG_SINGLE_CONF = 0.20

TWO_HIT_CONF_SUM = 0.10
TWO_HIT_MIN_IOU = 0.25


# ============================================================
# TRACK LIFETIME
# ============================================================

MAX_TENTATIVE_MISSES = 1

# GT experiment showed propagation is still very good at +5.
MAX_CONFIRMED_MISSES = 5


# ============================================================
# ASSOCIATION
# ============================================================

MIN_ASSOC_IOU = 0.03
MIN_ASSOC_CENTER_PX = 10.0
ASSOC_SCALE_FACTOR = 1.25


# Hard safety against exploding false tracks.
MAX_OUTPUT_TRACKS = 400


BBox = Tuple[
    float,
    float,
    float,
    float,
]


# ============================================================
# LOAD YOLO
# ============================================================

_detector = YOLO(
    MODEL_WEIGHTS
)

_dummy = np.zeros(
    (IMAGE_H, IMAGE_W, 3),
    dtype=np.uint8,
)

_detector.predict(
    _dummy,
    imgsz=960,
    conf=PROPOSAL_CONF,
    iou=NMS_IOU,
    max_det=MAX_DETECTIONS,
    device=DEVICE,
    half=USE_HALF,

    # Very important:
    # suppress overlapping boxes even when YOLO assigns
    # different classes.
    agnostic_nms=True,

    verbose=False,
)


# ============================================================
# BASIC GEOMETRY
# ============================================================

def bbox_area(
    bbox: BBox,
) -> float:

    x1, y1, x2, y2 = bbox

    return (
        max(0.0, x2 - x1)
        *
        max(0.0, y2 - y1)
    )


def bbox_center(
    bbox: BBox,
):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
    )


def bbox_iou(
    a: BBox,
    b: BBox,
) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    if union <= 0.0:
        return 0.0

    return inter / union


def clip_pixel_bbox(
    bbox: BBox,
) -> Optional[BBox]:

    x1, y1, x2, y2 = bbox

    x1 = max(
        0.0,
        min(
            float(IMAGE_W),
            x1,
        ),
    )

    x2 = max(
        0.0,
        min(
            float(IMAGE_W),
            x2,
        ),
    )

    y1 = max(
        0.0,
        min(
            float(IMAGE_H),
            y1,
        ),
    )

    y2 = max(
        0.0,
        min(
            float(IMAGE_H),
            y2,
        ),
    )

    if (
        x2 - x1 < 1.0
        or
        y2 - y1 < 1.0
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def affine_to_homogeneous(
    A: np.ndarray,
) -> np.ndarray:

    H = np.eye(
        3,
        dtype=np.float64,
    )

    H[:2, :] = A

    return H


def warp_bbox(
    bbox: BBox,
    H: np.ndarray,
) -> Optional[BBox]:

    x1, y1, x2, y2 = bbox

    corners = np.float32(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    try:

        warped = (
            cv2.perspectiveTransform(
                corners,
                H,
            )[:, 0, :]
        )

    except cv2.error:

        return None


    if not np.isfinite(
        warped
    ).all():

        return None


    result = (
        float(
            warped[:, 0].min()
        ),
        float(
            warped[:, 1].min()
        ),
        float(
            warped[:, 0].max()
        ),
        float(
            warped[:, 1].max()
        ),
    )

    return clip_pixel_bbox(
        result
    )


def blend_bbox(
    predicted: BBox,
    detected: BBox,
    conf: float,
) -> BBox:

    # Geometry carries most of the location.
    # YOLO gently corrects accumulated drift.
    alpha = min(
        0.65,
        max(
            0.25,
            0.25
            +
            2.0 * conf,
        ),
    )

    return tuple(
        (
            (1.0 - alpha) * p
            +
            alpha * d
        )
        for p, d
        in zip(
            predicted,
            detected,
        )
    )


def pixel_to_global(
    bbox: BBox,
):

    x1, y1, x2, y2 = bbox

    return (
        max(
            0.0,
            min(
                1.0,
                x1 / IMAGE_W,
            ),
        ),

        max(
            0.0,
            min(
                1.0,
                y1 / IMAGE_H,
            ),
        ),

        max(
            0.0,
            min(
                1.0,
                x2 / IMAGE_W,
            ),
        ),

        max(
            0.0,
            min(
                1.0,
                y2 / IMAGE_H,
            ),
        ),
    )


# ============================================================
# FAST LK + AFFINE
# ============================================================

def estimate_fast_motion(
    previous_gray,
    current_gray,
):

    # Detect corners only in previous frame.
    points0 = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=7,
        useHarrisDetector=False,
    )

    if (
        points0 is None
        or
        len(points0)
        <
        MIN_FLOW_POINTS
    ):

        return (
            None,
            0.0,
            0,
        )


    # Track those corners into current frame.
    points1, status, error = (
        cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points0,
            None,
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                |
                cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
    )

    if (
        points1 is None
        or
        status is None
    ):

        return (
            None,
            0.0,
            0,
        )


    valid = (
        status
        .reshape(-1)
        .astype(bool)
    )

    p0 = (
        points0
        .reshape(-1, 2)[valid]
    )

    p1 = (
        points1
        .reshape(-1, 2)[valid]
    )


    finite = (
        np.isfinite(p0).all(axis=1)
        &
        np.isfinite(p1).all(axis=1)
    )

    p0 = p0[finite]
    p1 = p1[finite]


    if (
        len(p0)
        <
        MIN_FLOW_POINTS
    ):

        return (
            None,
            0.0,
            len(p0),
        )


    # Partial affine:
    # translation + rotation + uniform scale.
    #
    # Helsinki test already showed this is extremely accurate.
    A, inliers = (
        cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=
                RANSAC_THRESHOLD,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
    )


    if (
        A is None
        or
        inliers is None
    ):

        return (
            None,
            0.0,
            len(p0),
        )


    inlier_ratio = float(
        inliers
        .reshape(-1)
        .mean()
    )


    if (
        inlier_ratio
        <
        MIN_AFFINE_INLIER
    ):

        return (
            None,
            inlier_ratio,
            len(p0),
        )


    return (
        affine_to_homogeneous(
            A
        ),
        inlier_ratio,
        len(p0),
    )


# ============================================================
# YOLO PROPOSALS
# ============================================================

def detect_proposals(
    image,
):

    results = _detector.predict(
        image,
        imgsz=960,
        conf=PROPOSAL_CONF,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        half=USE_HALF,

        # Merge spatially overlapping predictions
        # regardless of class.
        agnostic_nms=True,

        verbose=False,
    )


    detections = []


    if (
        not results
        or
        results[0].boxes
        is None
    ):

        return detections


    result = results[0]

    names = result.names


    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )


    confs = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )


    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )


    for (
        box,
        conf,
        cls_id,
    ) in zip(
        boxes,
        confs,
        classes,
    ):

        object_id = names[
            int(cls_id)
        ]


        if (
            object_id
            not in OBJECT_CLASSES
        ):

            continue


        bbox = clip_pixel_bbox(
            tuple(
                float(v)
                for v
                in box
            )
        )


        if bbox is None:
            continue


        detections.append(
            {
                "bbox":
                    bbox,

                "object_id":
                    object_id,

                "confidence":
                    float(conf),
            }
        )


    return detections


# ============================================================
# TRACK
# ============================================================

@dataclass
class Track:

    track_id: int

    bbox: BBox

    first_frame: int
    last_det_frame: int

    hits: int = 1
    misses: int = 0

    confirmed: bool = False

    ema_conf: float = 0.0
    max_conf: float = 0.0


    # One geometric trajectory can accumulate evidence
    # for several possible classes.
    class_scores: Dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )


    recent_frames: Deque[int] = field(
        default_factory=lambda:
            deque(
                maxlen=RECENT_WINDOW
            )
    )


    recent_confs: Deque[float] = field(
        default_factory=lambda:
            deque(
                maxlen=RECENT_WINDOW
            )
    )


    recent_match_ious: Deque[float] = field(
        default_factory=lambda:
            deque(
                maxlen=RECENT_WINDOW
            )
    )


    matched_this_frame: bool = False


    def best_class(
        self,
    ) -> str:

        if not self.class_scores:
            return ""

        return max(
            self.class_scores,
            key=self.class_scores.get,
        )


    def class_consistency(
        self,
    ) -> float:

        total = float(
            sum(
                self.class_scores.values()
            )
        )

        if total <= 0.0:
            return 0.0

        return float(
            max(
                self.class_scores.values()
            )
            /
            total
        )


    def recent_hit_count(
        self,
        frame: int,
    ) -> int:

        lower = (
            frame
            -
            (RECENT_WINDOW - 1)
        )

        return sum(
            1
            for f
            in self.recent_frames
            if f >= lower
        )


    def mean_recent_iou(
        self,
    ) -> float:

        if not self.recent_match_ious:
            return 0.0

        return float(
            np.mean(
                self.recent_match_ious
            )
        )


    def output_confidence(
        self,
    ) -> float:

        support = min(
            1.0,
            self.hits / 5.0,
        )


        consistency = (
            self.class_consistency()
        )


        if self.recent_match_ious:

            geometry = min(
                1.0,
                self.mean_recent_iou()
                /
                0.60,
            )

        else:

            geometry = 0.35


        detector_strength = min(
            1.0,

            max(
                self.ema_conf,
                self.max_conf * 0.6,
            )
            /
            0.12,
        )


        freshness = (
            0.78
            **
            self.misses
        )


        score = (
            0.10

            +
            0.30 * support

            +
            0.25 * consistency

            +
            0.20 * geometry

            +
            0.15 * detector_strength
        ) * freshness


        return float(
            max(
                0.005,
                min(
                    0.99,
                    score,
                ),
            )
        )


# ============================================================
# WORLD
# ============================================================

class World:

    def __init__(
        self,
    ):

        self.attempt_key = None

        self.prev_gray = None
        self.prev_frame = None

        self.next_track_id = 1

        self.tracks: Dict[
            int,
            Track,
        ] = {}


    def reset(
        self,
        attempt_key,
    ):

        self.attempt_key = (
            attempt_key
        )

        self.prev_gray = None
        self.prev_frame = None

        self.next_track_id = 1

        self.tracks = {}


    def create_track(
        self,
        det,
        frame,
    ):

        conf = float(
            det["confidence"]
        )


        track = Track(
            track_id=
                self.next_track_id,

            bbox=
                det["bbox"],

            first_frame=
                frame,

            last_det_frame=
                frame,

            hits=
                1,

            misses=
                0,

            confirmed=
                (
                    conf
                    >=
                    STRONG_SINGLE_CONF
                ),

            ema_conf=
                conf,

            max_conf=
                conf,

            class_scores={
                det["object_id"]:
                    math.sqrt(
                        max(
                            conf,
                            1e-8,
                        )
                    )
            },

            matched_this_frame=
                True,
        )


        track.recent_frames.append(
            frame
        )

        track.recent_confs.append(
            conf
        )


        self.tracks[
            track.track_id
        ] = track


        self.next_track_id += 1


        return track


_world = World()


# ============================================================
# CLASS VOTING
# ============================================================

def update_class_vote(
    track: Track,
    object_id: str,
    conf: float,
):

    # Slow decay so repeated later evidence can
    # overturn an early class mistake.
    for cls in list(
        track.class_scores
    ):

        track.class_scores[
            cls
        ] *= 0.99


    # sqrt gives useful weight even to hidden-domain
    # low-confidence detections.
    weight = math.sqrt(
        max(
            conf,
            1e-8,
        )
    )


    track.class_scores[
        object_id
    ] = (
        track.class_scores.get(
            object_id,
            0.0,
        )
        +
        weight
    )


# ============================================================
# ASSOCIATION
# ============================================================

def association_metrics(
    track_bbox: BBox,
    det_bbox: BBox,
):

    overlap = bbox_iou(
        track_bbox,
        det_bbox,
    )


    tx, ty = bbox_center(
        track_bbox
    )

    dx, dy = bbox_center(
        det_bbox
    )


    distance = math.hypot(
        dx - tx,
        dy - ty,
    )


    tw = (
        track_bbox[2]
        -
        track_bbox[0]
    )

    th = (
        track_bbox[3]
        -
        track_bbox[1]
    )

    dw = (
        det_bbox[2]
        -
        det_bbox[0]
    )

    dh = (
        det_bbox[3]
        -
        det_bbox[1]
    )


    scale = max(
        tw,
        th,
        dw,
        dh,
        1.0,
    )


    max_distance = max(
        MIN_ASSOC_CENTER_PX,

        ASSOC_SCALE_FACTOR
        *
        scale,
    )


    allowed = (
        overlap
        >=
        MIN_ASSOC_IOU

        or

        distance
        <=
        max_distance
    )


    if max_distance > 0:

        proximity = max(
            0.0,
            1.0
            -
            distance
            /
            max_distance,
        )

    else:

        proximity = 0.0


    score = (
        overlap
        +
        0.15
        *
        proximity
    )


    return (
        allowed,
        score,
        overlap,
        distance,
    )


# ============================================================
# CONFIRMATION
# ============================================================

def maybe_confirm(
    track: Track,
    frame: int,
):

    if track.confirmed:
        return


    # Very strong single detection.
    if (
        track.max_conf
        >=
        STRONG_SINGLE_CONF
    ):

        track.confirmed = True
        return


    recent_hits = (
        track.recent_hit_count(
            frame
        )
    )


    recent_conf_sum = float(
        sum(
            track.recent_confs
        )
    )


    mean_iou = (
        track.mean_recent_iou()
    )


    # Two reasonably strong consistent observations.
    if (
        recent_hits >= 2

        and

        recent_conf_sum
        >=
        TWO_HIT_CONF_SUM

        and

        mean_iou
        >=
        TWO_HIT_MIN_IOU
    ):

        track.confirmed = True
        return


    # Normal hidden-domain path:
    # 3 / 4 temporal consistency.
    if (
        recent_hits
        >=
        MIN_RECENT_HITS

        and

        recent_conf_sum
        >=
        MIN_RECENT_CONF_SUM

        and

        mean_iou
        >=
        MIN_CONFIRM_MATCH_IOU
    ):

        track.confirmed = True


# ============================================================
# WORLD UPDATE
# ============================================================

def update_world(
    frame,
    H,
    detections,
    had_previous_frame,
):

    for track in (
        _world.tracks.values()
    ):

        track.matched_this_frame = (
            False
        )


    # --------------------------------------------------------
    # 1. PROPAGATE TRACKS BY GLOBAL CAMERA MOTION
    # --------------------------------------------------------

    if H is not None:

        invalid = []

        for (
            track_id,
            track,
        ) in (
            _world.tracks.items()
        ):

            warped = warp_bbox(
                track.bbox,
                H,
            )

            if warped is None:

                invalid.append(
                    track_id
                )

            else:

                track.bbox = warped


        for track_id in invalid:

            del _world.tracks[
                track_id
            ]


    # If motion estimation failed, don't associate
    # detections against stale geometry.
    can_associate = (
        not had_previous_frame
        or
        H is not None
    )


    assigned_tracks = set()
    assigned_dets = set()


    # --------------------------------------------------------
    # 2. CLASS-AGNOSTIC GEOMETRIC ASSOCIATION
    # --------------------------------------------------------

    if (
        can_associate
        and
        _world.tracks
        and
        detections
    ):

        pairs = []


        for track in (
            _world.tracks.values()
        ):

            for (
                det_idx,
                det,
            ) in enumerate(
                detections
            ):

                (
                    allowed,
                    score,
                    overlap,
                    distance,
                ) = association_metrics(
                    track.bbox,
                    det["bbox"],
                )


                if allowed:

                    pairs.append(
                        (
                            score,
                            overlap,
                            -distance,
                            track.track_id,
                            det_idx,
                        )
                    )


        pairs.sort(
            reverse=True
        )


        for (
            score,
            overlap,
            neg_distance,
            track_id,
            det_idx,
        ) in pairs:

            if (
                track_id
                in assigned_tracks
            ):

                continue


            if (
                det_idx
                in assigned_dets
            ):

                continue


            track = (
                _world.tracks.get(
                    track_id
                )
            )


            if track is None:
                continue


            det = detections[
                det_idx
            ]


            conf = float(
                det["confidence"]
            )


            # Affine gives predicted geometry.
            # Detector only corrects it.
            track.bbox = blend_bbox(
                track.bbox,
                det["bbox"],
                conf,
            )


            track.last_det_frame = (
                frame
            )

            track.hits += 1
            track.misses = 0

            track.matched_this_frame = (
                True
            )


            track.ema_conf = (
                0.80
                *
                track.ema_conf

                +

                0.20
                *
                conf
            )


            track.max_conf = max(
                track.max_conf,
                conf,
            )


            track.recent_frames.append(
                frame
            )

            track.recent_confs.append(
                conf
            )

            track.recent_match_ious.append(
                float(
                    overlap
                )
            )


            update_class_vote(
                track,
                det["object_id"],
                conf,
            )


            maybe_confirm(
                track,
                frame,
            )


            assigned_tracks.add(
                track_id
            )

            assigned_dets.add(
                det_idx
            )


    # --------------------------------------------------------
    # 3. UNMATCHED TRACKS SURVIVE TEMPORARILY
    # --------------------------------------------------------

    for track in list(
        _world.tracks.values()
    ):

        if (
            track.track_id
            not in assigned_tracks
        ):

            track.misses += 1


    # --------------------------------------------------------
    # 4. NEW PROPOSALS
    # --------------------------------------------------------

    for (
        det_idx,
        det,
    ) in enumerate(
        detections
    ):

        if (
            det_idx
            not in assigned_dets
        ):

            _world.create_track(
                det,
                frame,
            )


    # --------------------------------------------------------
    # 5. PRUNE
    # --------------------------------------------------------

    stale = []


    for (
        track_id,
        track,
    ) in (
        _world.tracks.items()
    ):

        if track.confirmed:

            max_misses = (
                MAX_CONFIRMED_MISSES
            )

        else:

            max_misses = (
                MAX_TENTATIVE_MISSES
            )


        if (
            track.misses
            >
            max_misses
        ):

            stale.append(
                track_id
            )


    for track_id in stale:

        del _world.tracks[
            track_id
        ]


# ============================================================
# ATTEMPT
# ============================================================

def attempt_key_from_request(
    request,
):

    return str(
        request.request_id
    ).split(":")[0]


# ============================================================
# MAIN
# ============================================================

def predict(
    request:
        DroneFlybyPredictRequestDto,
) -> DroneFlybyPredictResponseDto:

    frame = int(
        request.frame
    )

    frame = int(request.frame)

    attempt_key = attempt_key_from_request(request)

    new_attempt = (
        _world.attempt_key != attempt_key
    )
    
    frame_restarted = (
        _world.prev_frame is not None
        and frame < _world.prev_frame
    )
    
    if new_attempt or frame_restarted:
    
        logger.warning(
            "RESET WORLD frame=%s prev_frame=%s "
            "new_attempt=%s frame_restarted=%s version=%s",
            frame,
            _world.prev_frame,
            new_attempt,
            frame_restarted,
            VERSION,
        )
    
        _world.reset(attempt_key)
    

    try:

        image = decode_view(
            request.view
        )


        # This experiment stays L0 only.
        if (
            int(
                request.view.resolution_level
            )
            !=
            0
        ):

            logger.warning(
                "Unexpected non-L0 "
                "frame=%s level=%s",
                frame,
                request.view.resolution_level,
            )


        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )


        had_previous_frame = (
            _world.prev_gray
            is not None
        )


        H = None
        inlier_ratio = 0.0
        flow_points = 0


        if had_previous_frame:

            (
                H,
                inlier_ratio,
                flow_points,
            ) = estimate_fast_motion(
                _world.prev_gray,
                gray,
            )


        detections = (
            detect_proposals(
                image
            )
        )


        update_world(
            frame=
                frame,

            H=
                H,

            detections=
                detections,

            had_previous_frame=
                had_previous_frame,
        )


        # ----------------------------------------------------
        # OUTPUT CONFIRMED TRACKS ONLY
        # ----------------------------------------------------

        confirmed = [
            track
            for track
            in _world.tracks.values()
            if track.confirmed
        ]


        # Highest-quality persistent tracks first.
        confirmed.sort(
            key=lambda t:
                t.output_confidence(),
            reverse=True,
        )


        # Safety cap.
        confirmed = (
            confirmed[
                :MAX_OUTPUT_TRACKS
            ]
        )


        annotations = []


        for track in confirmed:

            bbox = clip_pixel_bbox(
                track.bbox
            )


            if bbox is None:
                continue


            object_id = (
                track.best_class()
            )


            if (
                object_id
                not in OBJECT_CLASSES
            ):

                continue


            global_bbox = (
                pixel_to_global(
                    bbox
                )
            )


            annotations.append(
                DroneFlybyPredictionDto(
                    object_id=
                        object_id,

                    bbox=[
                        float(
                            global_bbox[0]
                        ),
                        float(
                            global_bbox[1]
                        ),
                        float(
                            global_bbox[2]
                        ),
                        float(
                            global_bbox[3]
                        ),
                    ],

                    confidence=
                        track.output_confidence(),
                )
            )


        logger.info(
            "frame=%s "
            "proposals=%s "
            "tracks=%s "
            "confirmed=%s "
            "outputs=%s "
            "affine=%s "
            "inlier=%.3f "
            "flow=%s",

            frame,

            len(
                detections
            ),

            len(
                _world.tracks
            ),

            len(
                [
                    t
                    for t
                    in _world.tracks.values()
                    if t.confirmed
                ]
            ),

            len(
                annotations
            ),

            (
                "ok"
                if H is not None
                else
                "none"
            ),

            inlier_ratio,

            flow_points,
        )


        _world.prev_gray = gray
        _world.prev_frame = frame


        return (
            DroneFlybyPredictResponseDto(
                request_id=
                    request.request_id,

                frame=
                    request.frame,

                annotations=
                    annotations,

                requested_view=
                    None,
            )
        )


    except Exception:

        logger.exception(
            "Prediction failed frame=%s",
            frame,
        )


        return (
            DroneFlybyPredictResponseDto(
                request_id=
                    request.request_id,

                frame=
                    request.frame,

                annotations=
                    [],

                requested_view=
                    None,
            )
        )
