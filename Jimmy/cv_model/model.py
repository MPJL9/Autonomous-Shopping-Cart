"""
DistanceTurnModel
=================
Uses a frozen YOLOv8-Pose backbone to extract per-frame features, then:

1. Distance   — calibrated formula: d = a / bbox_h_norm + b  (2 params, fit via least-squares)
2. Turn intent — simple linear classifier over a flattened window of pose angles

The YOLO backbone runs outside PyTorch (via ultralytics).
"""

import math

import numpy as np


# ── COCO-17 keypoint indices ────────────────────────────────────────────────
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_distance_features(bbox_xyxy, kpts_xy, kpts_conf, img_h,
                               conf_min=0.25):
    """
    Features for distance estimation.

    Returns dict with:
        bbox_h_norm  — bbox height / image height  (primary distance cue)
        torso_norm   — torso pixel length / image height (fallback cue)
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(float(y2 - y1), 1.0)

    feats = {"bbox_h_norm": bbox_h / img_h, "torso_norm": None}

    def ok(i):
        return kpts_conf[i] >= conf_min and np.isfinite(kpts_xy[i]).all()

    if ok(L_SHOULDER) and ok(R_SHOULDER) and ok(L_HIP) and ok(R_HIP):
        mid_sh = 0.5 * (kpts_xy[L_SHOULDER] + kpts_xy[R_SHOULDER])
        mid_hp = 0.5 * (kpts_xy[L_HIP] + kpts_xy[R_HIP])
        feats["torso_norm"] = float(np.linalg.norm(mid_sh - mid_hp)) / img_h

    return feats


def extract_turn_features(kpts_xy, kpts_conf, conf_min=0.25,
                          angular_vel=0.0, bbox_center_vel=0.0):
    """
    Features for turn-intent classification.

    Returns a 5-dim numpy array:
        [shoulder_angle, hip_angle, twist, angular_vel, bbox_center_vel]

    Parameters
    ----------
    angular_vel : float
        Change in shoulder_angle from the previous frame (radians/frame).
        Pass 0.0 for the first frame or when unavailable.
    bbox_center_vel : float
        Change in bbox center x from the previous frame (pixels/frame).
        Pass 0.0 for the first frame or when unavailable.

    All zero if keypoints are not confident enough.
    """
    feats = np.zeros(5, dtype=np.float32)

    def ok(i):
        return kpts_conf[i] >= conf_min and np.isfinite(kpts_xy[i]).all()

    if ok(L_SHOULDER) and ok(R_SHOULDER):
        dx = float(kpts_xy[R_SHOULDER][0] - kpts_xy[L_SHOULDER][0])
        dy = float(kpts_xy[R_SHOULDER][1] - kpts_xy[L_SHOULDER][1])
        feats[0] = math.atan2(dy, dx)   # shoulder angle

    if ok(L_HIP) and ok(R_HIP):
        dx = float(kpts_xy[R_HIP][0] - kpts_xy[L_HIP][0])
        dy = float(kpts_xy[R_HIP][1] - kpts_xy[L_HIP][1])
        feats[1] = math.atan2(dy, dx)   # hip angle

    feats[2] = feats[0] - feats[1]      # twist
    feats[3] = float(angular_vel)       # shoulder angle velocity (rad/frame)
    feats[4] = float(bbox_center_vel)   # bbox center x velocity (px/frame)

    return feats


NUM_TURN_FEATURES = 5  # per frame: [shoulder_angle, hip_angle, twist, angular_vel, bbox_center_vel]


# ── Angle estimation ──────────────────────────────────────────────────────────

def estimate_person_angle(bbox_xyxy, img_w, focal_length_x=None,
                          hfov_deg=60.0):
    """
    Estimate the horizontal angle (radians) of the person relative to the
    camera's optical axis.

    Positive = person is to the right of centre, negative = left.

    Parameters
    ----------
    bbox_xyxy : array-like
        Bounding box [x1, y1, x2, y2].
    img_w : int
        Image width in pixels.
    focal_length_x : float or None
        Focal length in pixels (from camera calibration).  If provided, this
        is used directly.  Otherwise the angle is approximated from
        ``hfov_deg``.
    hfov_deg : float
        Assumed horizontal field-of-view in degrees.  Only used when
        ``focal_length_x`` is None.

    Returns
    -------
    float
        Angle in radians.
    """
    x1, _, x2, _ = bbox_xyxy
    cx = 0.5 * (float(x1) + float(x2))
    offset_px = cx - img_w / 2.0

    if focal_length_x is not None:
        fx = float(focal_length_x)
    else:
        fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))

    return math.atan2(offset_px, fx)


# ── Distance estimator (calibrated formula, no neural net) ──────────────────

class CalibratedDistance:
    """
    d = a / bbox_h_norm + b

    Fit a and b from ArUco ground-truth data using least-squares.
    This is the pinhole model: distance is inversely proportional to the
    apparent height of the person in the image.

    Total parameters: 2
    """

    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b

    def predict(self, bbox_h_norm):
        """Predict distance in meters from normalized bbox height."""
        return self.a / max(bbox_h_norm, 1e-6) + self.b

    def fit(self, bbox_h_norms, gt_distances):
        """
        Fit a, b from arrays of bbox_h_norm and ground-truth distances.

        Uses least-squares on:  d = a * (1/h) + b
        """
        h = np.array(bbox_h_norms, dtype=np.float64)
        d = np.array(gt_distances, dtype=np.float64)

        # Design matrix: [1/h, 1]
        A = np.column_stack([1.0 / np.clip(h, 1e-6, None), np.ones(len(h))])
        # Solve A @ [a, b] ≈ d
        result, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
        self.a, self.b = float(result[0]), float(result[1])

    def save(self, path):
        np.savez(path, a=self.a, b=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(a=float(data["a"]), b=float(data["b"]))


# ── Turn-intent classifier (linear on flattened window) ─────────────────────

class TurnClassifier:
    """
    Flattens a window of [shoulder_angle, hip_angle, twist, angular_vel,
    bbox_center_vel] over T frames and runs logistic regression (softmax
    over 3 classes).

    Total parameters: (T * 5) * 3  +  3  (weights + bias)
    With T=15: 228 params

    Classes: 0=left, 1=right, 2=straight

    Note: this is a linear baseline. The target architecture (GRU / 1D-CNN)
    will be chosen after initial experiments.
    """

    def __init__(self, window_size=15, num_classes=3):
        self.T = window_size
        self.num_classes = num_classes
        in_dim = window_size * NUM_TURN_FEATURES
        self.W = np.zeros((in_dim, num_classes), dtype=np.float64)
        self.bias = np.zeros(num_classes, dtype=np.float64)

    def predict_probs(self, window):
        """
        window: (T, 3) numpy array of turn features.
        Returns: (3,) probability array [P(left), P(right), P(straight)].
        """
        x = window.flatten()
        logits = x @ self.W + self.bias
        # Softmax
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def predict(self, window):
        """Returns predicted class index."""
        return int(self.predict_probs(window).argmax())

    def fit(self, windows, labels, lr=0.01, epochs=200):
        """
        Simple gradient descent with softmax cross-entropy.

        windows: (N, T, 3) array
        labels:  (N,) int array  (0=left, 1=right, 2=straight)
        """
        N = len(labels)
        X = windows.reshape(N, -1)  # (N, T*3)
        Y = np.eye(self.num_classes)[labels]  # one-hot (N, 3)

        for epoch in range(epochs):
            logits = X @ self.W + self.bias           # (N, 3)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)  # (N, 3)

            # Cross-entropy loss
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

            # Gradients
            grad = (probs - Y) / N            # (N, 3)
            dW = X.T @ grad                   # (T*3, 3)
            db = grad.sum(axis=0)             # (3,)

            self.W -= lr * dW
            self.bias -= lr * db

            if (epoch + 1) % 50 == 0:
                acc = (probs.argmax(axis=1) == labels).mean()
                print(f"  epoch {epoch+1:3d}  loss={loss:.4f}  acc={acc:.3f}")

    def save(self, path):
        np.savez(path, W=self.W, bias=self.bias, T=self.T)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(window_size=int(data["T"]))
        obj.W = data["W"]
        obj.bias = data["bias"]
        return obj
