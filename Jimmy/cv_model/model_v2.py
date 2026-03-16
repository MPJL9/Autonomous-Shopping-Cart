"""
Model V2 — slightly more complex fallback
==========================================
Use this if the simple calibrated model (model.py) isn't accurate enough.

Differences from V1:
  Distance  — small MLP (1 hidden layer, 16 units) on 5 features instead of
              a 2-param formula on 1 feature. Handles crouching, partial
              occlusion, and varying body proportions better.
              Parameters: 5*16 + 16 + 16*1 + 1 = 113
  Turn      — 2-layer MLP on the flattened window instead of linear.
              Captures non-linear interactions between angles across frames.
              Parameters: (T*3)*16 + 16 + 16*3 + 3 = 787  (T=15)

Total trainable: ~900 params  (vs 140 in V1)

Still numpy-only, no PyTorch dependency.
"""

import math

import numpy as np

# ── COCO-17 keypoint indices ────────────────────────────────────────────────
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_distance_features_v2(bbox_xyxy, kpts_xy, kpts_conf, img_w, img_h,
                                  conf_min=0.25):
    """
    5-dim feature vector for distance.

        0  bbox_h_norm      — bbox height / image height
        1  bbox_w_norm      — bbox width  / image width
        2  bbox_area_norm   — bbox area / image area
        3  torso_len_norm   — torso pixel length / image height
        4  shoulder_w_norm  — shoulder width / image width
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_h = max(float(y2 - y1), 1.0)
    bbox_w = max(float(x2 - x1), 1.0)

    feats = np.zeros(5, dtype=np.float32)
    feats[0] = bbox_h / img_h
    feats[1] = bbox_w / img_w
    feats[2] = (bbox_h * bbox_w) / (img_h * img_w)

    def ok(i):
        return kpts_conf[i] >= conf_min and np.isfinite(kpts_xy[i]).all()

    if ok(L_SHOULDER) and ok(R_SHOULDER) and ok(L_HIP) and ok(R_HIP):
        mid_sh = 0.5 * (kpts_xy[L_SHOULDER] + kpts_xy[R_SHOULDER])
        mid_hp = 0.5 * (kpts_xy[L_HIP] + kpts_xy[R_HIP])
        feats[3] = float(np.linalg.norm(mid_sh - mid_hp)) / img_h

    if ok(L_SHOULDER) and ok(R_SHOULDER):
        feats[4] = float(np.linalg.norm(
            kpts_xy[R_SHOULDER] - kpts_xy[L_SHOULDER]
        )) / img_w

    return feats


NUM_DIST_FEATURES_V2 = 5


def extract_turn_features_v2(kpts_xy, kpts_conf, conf_min=0.25,
                              angular_vel=0.0, bbox_center_vel=0.0):
    """
    5-dim turn features:
        [shoulder_angle, hip_angle, twist, angular_vel, bbox_center_vel]

    Parameters
    ----------
    angular_vel : float
        Change in shoulder_angle from the previous frame (radians/frame).
    bbox_center_vel : float
        Change in bbox center x from the previous frame (pixels/frame).
    """
    feats = np.zeros(5, dtype=np.float32)

    def ok(i):
        return kpts_conf[i] >= conf_min and np.isfinite(kpts_xy[i]).all()

    if ok(L_SHOULDER) and ok(R_SHOULDER):
        dx = float(kpts_xy[R_SHOULDER][0] - kpts_xy[L_SHOULDER][0])
        dy = float(kpts_xy[R_SHOULDER][1] - kpts_xy[L_SHOULDER][1])
        feats[0] = math.atan2(dy, dx)

    if ok(L_HIP) and ok(R_HIP):
        dx = float(kpts_xy[R_HIP][0] - kpts_xy[L_HIP][0])
        dy = float(kpts_xy[R_HIP][1] - kpts_xy[L_HIP][1])
        feats[1] = math.atan2(dy, dx)

    feats[2] = feats[0] - feats[1]
    feats[3] = float(angular_vel)
    feats[4] = float(bbox_center_vel)
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


# ── Numpy MLP helpers ───────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0, x)


def _softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _init_weights(rows, cols):
    """He initialization."""
    std = math.sqrt(2.0 / rows)
    return np.random.randn(rows, cols).astype(np.float64) * std


# ── Distance MLP ────────────────────────────────────────────────────────────

class DistanceMLP:
    """
    5 features → Linear(5,16) → ReLU → Linear(16,1) → Softplus → distance

    Parameters: 5*16+16 + 16*1+1 = 113
    Trained with MSE via gradient descent.
    """

    def __init__(self, in_dim=NUM_DIST_FEATURES_V2, hidden=16):
        self.W1 = _init_weights(in_dim, hidden)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = _init_weights(hidden, 1)
        self.b2 = np.zeros(1, dtype=np.float64)

    def predict(self, x):
        """x: (5,) → scalar distance in meters."""
        h = _relu(x @ self.W1 + self.b1)
        return float(_softplus(h @ self.W2 + self.b2))

    def _forward_batch(self, X):
        """X: (N, 5) → (z1, h1, out) for backprop."""
        z1 = X @ self.W1 + self.b1        # (N, hidden)
        h1 = _relu(z1)                     # (N, hidden)
        z2 = h1 @ self.W2 + self.b2       # (N, 1)
        out = _softplus(z2)                # (N, 1)
        return z1, h1, z2, out

    def fit(self, X, y, lr=1e-3, epochs=300, batch_size=32):
        """
        X: (N, 5) features
        y: (N,) ground-truth distances
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)
        N = len(X)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(N)
            X, y = X[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                Xb = X[start:start+batch_size]
                yb = y[start:start+batch_size]
                B = len(Xb)

                z1, h1, z2, out = self._forward_batch(Xb)

                # MSE loss
                diff = out - yb                         # (B, 1)
                loss = float(np.mean(diff ** 2))
                epoch_loss += loss * B

                # Backprop
                d_out = 2 * diff / B                    # (B, 1)

                # Softplus gradient: sigmoid(z2)
                sig = 1.0 / (1.0 + np.exp(-z2))
                d_z2 = d_out * sig                      # (B, 1)

                dW2 = h1.T @ d_z2                       # (hidden, 1)
                db2 = d_z2.sum(axis=0)                  # (1,)

                d_h1 = d_z2 @ self.W2.T                 # (B, hidden)
                d_z1 = d_h1 * (z1 > 0).astype(float)   # ReLU grad

                dW1 = Xb.T @ d_z1                       # (5, hidden)
                db1 = d_z1.sum(axis=0)                  # (hidden,)

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / N
                print(f"  epoch {epoch+1:3d}  mse={avg_loss:.6f}")

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls()
        obj.W1, obj.b1 = d["W1"], d["b1"]
        obj.W2, obj.b2 = d["W2"], d["b2"]
        return obj


# ── Turn-intent 2-layer MLP ─────────────────────────────────────────────────

class TurnClassifierV2:
    """
    Flattened window → Linear(T*5, 16) → ReLU → Linear(16, 3) → softmax

    Parameters: (T*5)*16 + 16 + 16*3 + 3 = 1267  (T=15)
    Classes: 0=left, 1=right, 2=straight
    """

    def __init__(self, window_size=15, hidden=16, num_classes=3):
        self.T = window_size
        self.num_classes = num_classes
        in_dim = window_size * NUM_TURN_FEATURES

        self.W1 = _init_weights(in_dim, hidden)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = _init_weights(hidden, num_classes)
        self.b2 = np.zeros(num_classes, dtype=np.float64)

    def predict_probs(self, window):
        """window: (T, 3) → (3,) probabilities."""
        x = window.flatten().astype(np.float64)
        h = _relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return _softmax(logits)

    def predict(self, window):
        return int(self.predict_probs(window).argmax())

    def fit(self, windows, labels, lr=0.01, epochs=300, batch_size=32):
        """
        windows: (N, T, 3)
        labels:  (N,) int
        """
        N = len(labels)
        X = windows.reshape(N, -1).astype(np.float64)
        Y = np.eye(self.num_classes)[labels]  # one-hot

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            X, Y, labels = X[perm], Y[perm], labels[perm]

            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                Xb = X[start:start+batch_size]
                Yb = Y[start:start+batch_size]
                B = len(Xb)

                # Forward
                z1 = Xb @ self.W1 + self.b1            # (B, hidden)
                h1 = _relu(z1)
                logits = h1 @ self.W2 + self.b2        # (B, 3)

                # Softmax
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / exp.sum(axis=1, keepdims=True)

                # Cross-entropy loss
                loss = -np.mean(np.sum(Yb * np.log(probs + 1e-12), axis=1))
                epoch_loss += loss * B

                # Backprop
                d_logits = (probs - Yb) / B             # (B, 3)

                dW2 = h1.T @ d_logits
                db2 = d_logits.sum(axis=0)

                d_h1 = d_logits @ self.W2.T
                d_z1 = d_h1 * (z1 > 0).astype(float)

                dW1 = Xb.T @ d_z1
                db1 = d_z1.sum(axis=0)

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            if (epoch + 1) % 50 == 0:
                # Full-batch accuracy
                z1 = X @ self.W1 + self.b1
                h1 = _relu(z1)
                logits = h1 @ self.W2 + self.b2
                preds = logits.argmax(axis=1)
                acc = (preds == labels).mean()
                print(f"  epoch {epoch+1:3d}  loss={epoch_loss/N:.4f}  acc={acc:.3f}")

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, T=self.T)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(window_size=int(d["T"]))
        obj.W1, obj.b1 = d["W1"], d["b1"]
        obj.W2, obj.b2 = d["W2"], d["b2"]
        return obj
