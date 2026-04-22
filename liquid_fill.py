import numpy as np
import cv2
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple

def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8)


def get_pca_info(mask: np.ndarray):
    """PCA 主轴信息"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        h, w = mask.shape
        return 0.0, 1.0, np.array([w / 2, h / 2], dtype=np.float32)
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(coords, mean=None)
    angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    aspect = float(np.sqrt(max(eigenvalues[0, 0], 1e-6) /
                           max(eigenvalues[1, 0], 1e-6)))
    return float(angle), aspect, mean[0]

def find_longest_radial_direction(mask: np.ndarray,
                                    method: str = "pca") -> Tuple[float, np.ndarray]:
    """
    找到蒙版的最长径向方向
    
    返回: (angle_degrees, center)
        angle_degrees: 最长方向的角度(度)，以x轴正方向为0度，逆时针为正
        center: 蒙版质心 (x, y)
    
    method:
        "pca":       用PCA主轴 (快，对椭圆类形状好)
        "farthest":  找轮廓上最远点对 (精确反映"最长跨度")
        "furthest_from_center": 从质心出发最远的轮廓点方向
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        h, w = mask.shape
        return 0.0, np.array([w / 2, h / 2], dtype=np.float32)

    center = np.array([xs.mean(), ys.mean()], dtype=np.float32)

    if method == "pca":
        angle, _, center = get_pca_info(mask)
        return angle, center

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, center
    cnt = max(contours, key=cv2.contourArea).squeeze(axis=1)
    if cnt.ndim == 1:
        cnt = cnt.reshape(-1, 2)

    if method == "farthest":
        hull = cv2.convexHull(cnt).squeeze(axis=1)
        if hull.ndim == 1:
            hull = hull.reshape(-1, 2)

        max_dist = 0
        best_pair = (hull[0], hull[0])
        n = len(hull)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sum((hull[i] - hull[j]) ** 2)
                if d > max_dist:
                    max_dist = d
                    best_pair = (hull[i], hull[j])

        p1, p2 = best_pair
        direction = p2 - p1
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        center = ((p1 + p2) / 2).astype(np.float32)
        return float(angle), center

    elif method == "furthest_from_center":
        diffs = cnt - center
        dists = np.sum(diffs ** 2, axis=1)
        far_idx = np.argmax(dists)
        far_pt = cnt[far_idx]
        direction = far_pt - center
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        return float(angle), center

    else:
        raise ValueError(f"Unknown method: {method}")


def rotate_image_and_mask(img: np.ndarray,
                           mask: np.ndarray,
                           angle_deg: float,
                           center: Tuple[float, float],
                           out_size: Optional[Tuple[int, int]] = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    绕指定中心旋转图像和蒙版，输出尺寸扩展以免裁剪
    返回: (旋转后图像, 旋转后蒙版, 反向变换矩阵2x3)
    """
    h, w = mask.shape
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    if out_size is None:
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        out_size = (new_w, new_h)

    rotated_img = cv2.warpAffine(img, M, out_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, out_size,
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    M_full = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_full)[:2]

    return rotated_img, rotated_mask, M_inv


def unrotate_image(img: np.ndarray,
                    M_inv: np.ndarray,
                    out_size: Tuple[int, int]) -> np.ndarray:
    """用反向矩阵将图像旋转回原坐标系"""
    return cv2.warpAffine(img, M_inv, out_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)

def extract_ordered_contour(mask: np.ndarray, num_points: int = 300) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("空蒙版")
    cnt = max(contours, key=cv2.contourArea).squeeze(axis=1)
    if cnt.ndim == 1:
        cnt = cnt.reshape(-1, 2)

    dists = np.sqrt(np.sum(np.diff(cnt, axis=0, append=cnt[:1]) ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < 1e-6:
        return cnt.astype(np.float32)

    sample_t = np.linspace(0, total, num_points, endpoint=False)
    sampled = np.zeros((num_points, 2), dtype=np.float32)
    for i, t in enumerate(sample_t):
        idx = np.searchsorted(cum, t) - 1
        idx = np.clip(idx, 0, len(cnt) - 1)
        seg_len = cum[idx + 1] - cum[idx] if idx + 1 < len(cum) else 1.0
        alpha = (t - cum[idx]) / max(seg_len, 1e-6)
        p1 = cnt[idx]
        p2 = cnt[(idx + 1) % len(cnt)]
        sampled[i] = p1 * (1 - alpha) + p2 * alpha
    return sampled


def find_best_rotation_offset(src_pts, tgt_pts):
    n = len(src_pts)
    src_c = src_pts - src_pts.mean(axis=0)
    tgt_c = tgt_pts - tgt_pts.mean(axis=0)
    src_scale = np.sqrt((src_c ** 2).sum() / n) + 1e-8
    tgt_scale = np.sqrt((tgt_c ** 2).sum() / n) + 1e-8
    src_n = src_c / src_scale
    tgt_n = tgt_c / tgt_scale

    best_cost = np.inf
    best_offset, best_reverse = 0, False
    for k in range(n):
        rolled = np.roll(src_n, k, axis=0)
        cost = np.sum((rolled - tgt_n) ** 2)
        if cost < best_cost:
            best_cost, best_offset, best_reverse = cost, k, False
    src_rev = src_n[::-1]
    for k in range(n):
        rolled = np.roll(src_rev, k, axis=0)
        cost = np.sum((rolled - tgt_n) ** 2)
        if cost < best_cost:
            best_cost, best_offset, best_reverse = cost, k, True
    return best_offset, best_reverse


def match_boundaries(src_mask, tgt_mask, num_points=300):
    src_pts = extract_ordered_contour(src_mask, num_points)
    tgt_pts = extract_ordered_contour(tgt_mask, num_points)
    offset, reverse = find_best_rotation_offset(src_pts, tgt_pts)
    if reverse:
        src_pts = src_pts[::-1]
    src_pts = np.roll(src_pts, offset, axis=0)
    return src_pts, tgt_pts


def estimate_similarity_transform(src_pts: np.ndarray,
                                  tgt_pts: np.ndarray) -> np.ndarray:
    """
    估计 src -> tgt 的相似变换（旋转 + 平移 + 等比缩放）
    若估计失败，回退为单位变换
    """
    M, _ = cv2.estimateAffinePartial2D(
        src_pts.astype(np.float32),
        tgt_pts.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)
    return M.astype(np.float32)


def invert_affine_transform(M: np.ndarray) -> np.ndarray:
    M_full = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    return np.linalg.inv(M_full)[:2].astype(np.float32)


def transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts.astype(np.float32),
                       np.ones((len(pts), 1), dtype=np.float32)])
    transformed = pts_h @ M.T
    return transformed.astype(np.float32)


def normalize_angle_deg(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle

def rigid_prealign(source_img, source_mask, target_mask, canvas_shape,
                   allow_rotation: bool = True):
    """体积比缩放 + 主轴对齐，可选禁用额外旋转"""
    t_angle, _, t_center = get_pca_info(target_mask)
    s_angle, _, s_center = get_pca_info(source_mask)
    scale = np.sqrt(target_mask.sum() / max(source_mask.sum(), 1)) * 1.15

    h_c, w_c = canvas_shape
    best_overlap, best = -1, None
    angle_offsets = [0, 180] if allow_rotation else [0]
    for angle_offset in angle_offsets:
        angle = t_angle - s_angle + angle_offset
        if not allow_rotation:
            angle = 0.0
        M = cv2.getRotationMatrix2D(
            (float(s_center[0]), float(s_center[1])), angle, scale)
        M[0, 2] += t_center[0] - s_center[0]
        M[1, 2] += t_center[1] - s_center[1]
        w_img = cv2.warpAffine(source_img, M, (w_c, h_c),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        w_mask = cv2.warpAffine(source_mask, M, (w_c, h_c),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        w_img[w_mask == 0] = 0
        overlap = np.logical_and(w_mask > 0, target_mask > 0).sum()
        if overlap > best_overlap:
            best_overlap, best = overlap, (w_img, w_mask)
    return best

def apply_gravity_sink(boundary_pts: np.ndarray,
                        target_mask: np.ndarray,
                        strength: float = 0.3) -> np.ndarray:
    """
    在已经旋转到"最长方向垂直"后，重力恒为 (0, 1) 向下
    让边界点模拟液体下沉：
    下方的点被进一步拉下（液体堆积在底部，边界被拉伸）
    上方的点被压缩（液面趋于水平）
    """
    ys, xs = np.where(target_mask > 0)
    if len(ys) == 0:
        return boundary_pts

    y_min, y_max = ys.min(), ys.max()
    cy = (y_min + y_max) / 2

    shifted = boundary_pts.copy()
    for i, p in enumerate(boundary_pts):
        y_rel = p[1] - cy
        if y_rel < 0:
            shifted[i, 1] = p[1] + abs(y_rel) * strength
        else:
            shifted[i, 1] = p[1] + y_rel * strength * 0.3
    return shifted


def apply_liquid_level(boundary_pts: np.ndarray,
                        target_mask: np.ndarray,
                        fill_ratio: float = 1.0) -> np.ndarray:
    """
    模拟"液体没装满"：上方多余的空间被切掉，边界被压到液面高度
    fill_ratio=1.0 表示装满整个蒙版, 0.5 表示只装一半体积
    """
    if fill_ratio >= 1.0:
        return boundary_pts

    ys, xs = np.where(target_mask > 0)
    if len(ys) == 0:
        return boundary_pts

    total = len(ys)
    target_volume = total * fill_ratio
    y_sorted = np.sort(ys)[::-1]
    cumulative = np.arange(1, len(y_sorted) + 1)
    idx = np.searchsorted(cumulative, target_volume)
    idx = min(idx, len(y_sorted) - 1)
    liquid_surface_y = y_sorted[idx]
    shifted = boundary_pts.copy()
    shifted[:, 1] = np.maximum(shifted[:, 1], liquid_surface_y)
    return shifted

def solve_harmonic_map(region_mask, boundary_src, boundary_tgt):
    """
    给定区域内边界点的目标位置，解 Laplace 方程得到区域内所有点的映射
    返回 cv2.remap 用的 (map_x, map_y)
    """
    h, w = region_mask.shape
    ys, xs = np.where(region_mask > 0)
    n = len(ys)
    idx_map = -np.ones((h, w), dtype=np.int64)
    idx_map[ys, xs] = np.arange(n)

    region_coords = np.column_stack([ys, xs])
    tree = cKDTree(region_coords)

    fixed = {}
    for (sx, sy), (tx, ty) in zip(boundary_src, boundary_tgt):
        by, bx = int(round(sy)), int(round(sx))
        if 0 <= by < h and 0 <= bx < w and region_mask[by, bx]:
            fixed[(by, bx)] = (float(tx), float(ty))
        else:
            _, nn = tree.query([by, bx])
            ny, nx = region_coords[nn]
            fixed[(int(ny), int(nx))] = (float(tx), float(ty))

    A = lil_matrix((n, n))
    bx_v = np.zeros(n)
    by_v = np.zeros(n)
    for k in range(n):
        y, x = ys[k], xs[k]
        if (y, x) in fixed:
            A[k, k] = 1.0
            tx, ty = fixed[(y, x)]
            bx_v[k], by_v[k] = tx, ty
        else:
            cc = 0
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and region_mask[ny, nx]:
                    A[k, idx_map[ny, nx]] = -1.0
                    cc += 1
            if cc == 0:
                A[k, k] = 1.0
                bx_v[k], by_v[k] = x, y
            else:
                A[k, k] = cc

    A = csr_matrix(A)
    fx = spsolve(A, bx_v)
    fy = spsolve(A, by_v)

    map_x = np.full((h, w), -1, dtype=np.float32)
    map_y = np.full((h, w), -1, dtype=np.float32)
    map_x[ys, xs] = fx.astype(np.float32)
    map_y[ys, xs] = fy.astype(np.float32)
    return map_x, map_y


def clamp_map_to_source_mask(map_x: np.ndarray,
                             map_y: np.ndarray,
                             source_mask: np.ndarray,
                             target_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将落到源蒙版外的采样坐标钳制到最近的源蒙版内像素
    这样可以保证输出只使用源蒙版内部内容，同时尽量填满目标区域
    """
    valid_src_ys, valid_src_xs = np.where(source_mask > 0)
    if len(valid_src_ys) == 0:
        return map_x, map_y

    clamped_x = map_x.copy()
    clamped_y = map_y.copy()

    tgt_ys, tgt_xs = np.where(target_mask > 0)
    if len(tgt_ys) == 0:
        return clamped_x, clamped_y

    src_h, src_w = source_mask.shape
    sample_x = clamped_x[tgt_ys, tgt_xs]
    sample_y = clamped_y[tgt_ys, tgt_xs]

    inside = (
        (sample_x >= 0) & (sample_x <= src_w - 1) &
        (sample_y >= 0) & (sample_y <= src_h - 1)
    )
    rounded_x = np.clip(np.rint(sample_x).astype(np.int32), 0, src_w - 1)
    rounded_y = np.clip(np.rint(sample_y).astype(np.int32), 0, src_h - 1)
    on_mask = np.zeros_like(inside, dtype=bool)
    on_mask[inside] = source_mask[rounded_y[inside], rounded_x[inside]] > 0
    invalid = ~inside | ~on_mask
    if not np.any(invalid):
        return clamped_x, clamped_y

    valid_coords = np.column_stack([valid_src_xs, valid_src_ys]).astype(np.float32)
    tree = cKDTree(valid_coords)
    query = np.column_stack([sample_x[invalid], sample_y[invalid]]).astype(np.float32)
    _, nn_idx = tree.query(query, k=1)
    nearest = valid_coords[nn_idx]
    clamped_x[tgt_ys[invalid], tgt_xs[invalid]] = nearest[:, 0]
    clamped_y[tgt_ys[invalid], tgt_xs[invalid]] = nearest[:, 1]
    return clamped_x, clamped_y


def rasterize_boundary_mask(boundary_pts: np.ndarray,
                            shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(boundary_pts) == 0:
        return mask
    cv2.fillPoly(mask, [boundary_pts.reshape(-1, 1, 2).astype(np.int32)], 1)
    return mask


def get_row_spans(mask: np.ndarray) -> list[tuple[int, int, int]]:
    spans = []
    h, _ = mask.shape
    for y in range(h):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        spans.append((y, int(xs.min()), int(xs.max())))
    return spans


def solve_stretch_map(source_mask: np.ndarray,
                      target_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于纵向累计面积 + 行内归一化宽度的单调拉伸映射
    目标是尽量保留源蒙版内全部内容，只改变相对距离
    """
    h, w = target_mask.shape
    map_x = np.full((h, w), -1, dtype=np.float32)
    map_y = np.full((h, w), -1, dtype=np.float32)

    src_spans = get_row_spans(source_mask)
    tgt_spans = get_row_spans(target_mask)
    if not src_spans or not tgt_spans:
        return map_x, map_y

    src_rows = np.array([row for row, _, _ in src_spans], dtype=np.int32)
    src_left = np.array([left for _, left, _ in src_spans], dtype=np.float32)
    src_right = np.array([right for _, _, right in src_spans], dtype=np.float32)
    src_width = src_right - src_left + 1.0

    tgt_rows = np.array([row for row, _, _ in tgt_spans], dtype=np.int32)
    tgt_left = np.array([left for _, left, _ in tgt_spans], dtype=np.float32)
    tgt_right = np.array([right for _, _, right in tgt_spans], dtype=np.float32)
    tgt_width = tgt_right - tgt_left + 1.0

    src_cum_bottom = np.cumsum(src_width[::-1])[::-1]
    tgt_cum_bottom = np.cumsum(tgt_width[::-1])[::-1]
    src_total = float(src_cum_bottom[0])
    tgt_total = float(tgt_cum_bottom[0])
    if src_total <= 0 or tgt_total <= 0:
        return map_x, map_y

    src_rank = src_cum_bottom / src_total
    tgt_rank = tgt_cum_bottom / tgt_total

    src_rank_asc = src_rank[::-1]
    src_rows_asc = src_rows[::-1].astype(np.float32)
    src_left_asc = src_left[::-1]
    src_right_asc = src_right[::-1]

    row_to_src_y = np.interp(tgt_rank[::-1], src_rank_asc, src_rows_asc)[::-1]
    row_to_src_left = np.interp(tgt_rank[::-1], src_rank_asc, src_left_asc)[::-1]
    row_to_src_right = np.interp(tgt_rank[::-1], src_rank_asc, src_right_asc)[::-1]

    for idx, y_t in enumerate(tgt_rows):
        left_t = int(tgt_left[idx])
        right_t = int(tgt_right[idx])
        width_t = max(right_t - left_t + 1, 1)
        xs_t = np.arange(left_t, right_t + 1, dtype=np.float32)
        if width_t == 1:
            u = np.zeros_like(xs_t)
        else:
            u = (xs_t - left_t) / (width_t - 1)

        left_s = row_to_src_left[idx]
        right_s = row_to_src_right[idx]
        xs_s = left_s + u * max(right_s - left_s, 0.0)
        ys_s = np.full_like(xs_s, row_to_src_y[idx], dtype=np.float32)

        map_x[y_t, left_t:right_t + 1] = xs_s
        map_y[y_t, left_t:right_t + 1] = ys_s

    return map_x, map_y

def liquid_fill_with_auto_gravity(target_mask: np.ndarray,
                                    source_img: np.ndarray,
                                    source_mask: Optional[np.ndarray] = None,
                                    target_gravity_angle: Optional[float] = None,
                                    source_gravity_angle: Optional[float] = None,
                                    mapping_mode: str = "stretch",
                                    radial_method: str = "farthest",
                                    gravity_strength: float = 0.3,
                                    geometry_preservation: float = 0.75,
                                    fill_ratio: float = 1.0,
                                    num_boundary_points: int = 300,
                                    background_color=(0, 0, 0)
                                    ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    带自动重力方向识别的液体填充
    参数:
        target_mask:  目标蒙版 (H, W)
        source_img:   源图像 (H', W', 3)
        source_mask:  源蒙版 (H', W'), 可选
        target_gravity_angle: 目标蒙版中“上端在上、下端在下”的方向角度（度）
            None: 自动从目标蒙版估计
            其余值: 使用手动指定方向
        source_gravity_angle: 源图中“重力向下”方向的角度（度）
            None: 自动从源蒙版主轴估计
            其余值: 使用手动指定方向进行对齐
        mapping_mode: 变形模式
            "stretch": 单调拉伸，内容保留优先
            "harmonic": 调和映射，贴轮廓优先
        radial_method: 识别最长径向的方法
            "pca":      PCA 主轴
            "farthest": 凸包最远点对
            "furthest_from_center": 质心到轮廓最远点
        gravity_strength: 重力强度 [0, 1]，0=无重力
        geometry_preservation: 几何保真强度 [0, 1]
            0: 更贴目标边界，但源图内部结构更容易扭曲
            1: 完全趋向全局相似变换，内部几何关系最稳定
        fill_ratio: 液体占体积比 [0, 1]，1=装满
        num_boundary_points: 边界采样点数
    
    返回:
        (result_img, final_mask, info)
        info 包含调试信息: detected_angle, center 等
    """
    target_mask = ensure_binary_mask(target_mask)
    geometry_preservation = float(np.clip(geometry_preservation, 0.0, 1.0))
    if source_mask is None:
        source_mask = np.ones(source_img.shape[:2], dtype=np.uint8)
    else:
        source_mask = ensure_binary_mask(source_mask)

    H, W = target_mask.shape
    long_angle, long_center = find_longest_radial_direction(
        target_mask, method=radial_method)
    effective_target_gravity_angle = long_angle
    if target_gravity_angle is not None:
        effective_target_gravity_angle = normalize_angle_deg(target_gravity_angle)

    rotate_angle = effective_target_gravity_angle - 90
    diag = int(np.sqrt(H ** 2 + W ** 2)) + 20
    rotated_tgt, _, M_inv_tgt = rotate_image_and_mask(
        target_mask, target_mask,
        rotate_angle,
        (long_center[0], long_center[1]),
        out_size=(diag, diag))
    if target_gravity_angle is None:
        ys_r, _ = np.where(rotated_tgt > 0)
        if len(ys_r) > 0:
            cy = (ys_r.min() + ys_r.max()) / 2
            upper = (ys_r < cy).sum()
            lower = (ys_r >= cy).sum()
            if upper > lower:
                rotate_angle += 180
                rotated_tgt, _, M_inv_tgt = rotate_image_and_mask(
                    target_mask, target_mask,
                    rotate_angle,
                    (long_center[0], long_center[1]),
                    out_size=(diag, diag))

    s_angle, _, s_center = get_pca_info(source_mask)
    effective_source_gravity_angle = s_angle
    if source_gravity_angle is not None:
        effective_source_gravity_angle = normalize_angle_deg(source_gravity_angle)
    src_rotate = effective_source_gravity_angle - 90

    rotated_src_img, rotated_src_mask, _ = rotate_image_and_mask(
        source_img, source_mask, src_rotate,
        (s_center[0], s_center[1]),
        out_size=None)

    manual_gravity_locked = (
        target_gravity_angle is not None or source_gravity_angle is not None
    )
    pre_img, pre_mask = rigid_prealign(
        rotated_src_img, rotated_src_mask, rotated_tgt,
        (diag, diag), allow_rotation=not manual_gravity_locked)
    
    src_boundary, tgt_boundary = match_boundaries(
        pre_mask, rotated_tgt, num_boundary_points)

    if gravity_strength > 0:
        tgt_boundary = apply_gravity_sink(
            tgt_boundary, rotated_tgt, gravity_strength)

    if fill_ratio < 1.0:
        tgt_boundary = apply_liquid_level(
            tgt_boundary, rotated_tgt, fill_ratio)

    deformed_tgt_mask = rasterize_boundary_mask(tgt_boundary, rotated_tgt.shape)
    if np.any(deformed_tgt_mask):
        rotated_fill_mask = deformed_tgt_mask
    else:
        rotated_fill_mask = rotated_tgt

    if geometry_preservation > 0:
        similarity_M = estimate_similarity_transform(src_boundary, tgt_boundary)
        similarity_inv = invert_affine_transform(similarity_M)
        rigid_src_boundary = transform_points(tgt_boundary, similarity_inv)
        src_boundary = (
            (1.0 - geometry_preservation) * src_boundary +
            geometry_preservation * rigid_src_boundary
        ).astype(np.float32)

    if mapping_mode == "stretch":
        map_x, map_y = solve_stretch_map(pre_mask, rotated_fill_mask)
    elif mapping_mode == "harmonic":
        map_x, map_y = solve_harmonic_map(
            rotated_fill_mask, tgt_boundary, src_boundary)
    else:
        raise ValueError(f"Unknown mapping_mode: {mapping_mode}")
    map_x, map_y = clamp_map_to_source_mask(
        map_x, map_y, pre_mask, rotated_fill_mask
    )

    filled_rotated = cv2.remap(pre_img, map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    sampled_source_mask = cv2.remap(
        (pre_mask > 0).astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (map_x >= 0) & (rotated_fill_mask > 0) & (sampled_source_mask > 0)
    if fill_ratio < 1.0:
        pass

    out_rotated = np.zeros_like(filled_rotated)
    out_rotated[valid] = filled_rotated[valid]
    final_img = unrotate_image(out_rotated, M_inv_tgt, (W, H))
    final_mask_rotated = valid.astype(np.uint8) * 255
    final_mask = cv2.warpAffine(final_mask_rotated, M_inv_tgt, (W, H),
                                  flags=cv2.INTER_NEAREST)
    result = np.full((H, W, 3), background_color, dtype=np.uint8)
    result[final_mask > 0] = final_img[final_mask > 0]

    info = {
        "detected_long_angle": long_angle,
        "target_gravity_angle": effective_target_gravity_angle,
        "target_gravity_mode": (
            "manual" if target_gravity_angle is not None else "auto"
        ),
        "applied_rotation": rotate_angle,
        "long_center": long_center.tolist(),
        "source_gravity_angle": effective_source_gravity_angle,
        "source_gravity_mode": (
            "manual" if source_gravity_angle is not None else "auto"
        ),
        "mapping_mode": mapping_mode,
        "geometry_preservation": geometry_preservation,
    }
    return result, final_mask, info
