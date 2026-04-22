from __future__ import annotations

from typing import Any

import cv2
import gradio as gr
import numpy as np

from liquid_fill import liquid_fill_with_auto_gravity


def _to_rgb(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()


def _merge_editor_mask(editor_value: dict[str, Any] | None) -> np.ndarray | None:
    if not editor_value:
        return None

    background = editor_value.get("background")
    if background is None:
        return None

    height, width = background.shape[:2]
    merged = np.zeros((height, width), dtype=np.uint8)

    for layer in editor_value.get("layers") or []:
        if layer is None:
            continue
        if layer.ndim == 2:
            alpha = layer > 0
        elif layer.shape[2] == 4:
            alpha = layer[:, :, 3] > 0
        else:
            alpha = np.any(layer > 0, axis=2)
        merged[alpha] = 255

    return merged


def _make_mask_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    overlay = np.zeros_like(preview)
    overlay[:, :, 0] = 255
    active = mask > 0
    preview[active] = cv2.addWeighted(preview, 0.35, overlay, 0.65, 0)[active]
    return preview


def _normalize_angle_deg(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def _draw_gravity_overlay(
    image: np.ndarray | None,
    points: list[list[int]] | list[tuple[int, int]] | None,
) -> np.ndarray | None:
    if image is None:
        return None

    preview = image.copy()
    pts = [tuple(map(int, p)) for p in (points or [])]

    if len(pts) >= 1:
        cv2.circle(preview, pts[0], 8, (255, 255, 0), -1)
        cv2.putText(
            preview,
            "1",
            (pts[0][0] + 10, pts[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
    if len(pts) >= 2:
        cv2.arrowedLine(preview, pts[0], pts[1], (0, 255, 255), 3, tipLength=0.15)
        cv2.circle(preview, pts[1], 8, (0, 255, 255), -1)
        cv2.putText(
            preview,
            "2",
            (pts[1][0] + 10, pts[1][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    return preview


def _source_background(source_editor: dict[str, Any] | None) -> np.ndarray | None:
    return _to_rgb((source_editor or {}).get("background"))


def _target_background(target_editor: dict[str, Any] | None) -> np.ndarray | None:
    return _to_rgb((target_editor or {}).get("background"))


def _sync_gravity_preview(image: np.ndarray | None, name: str):
    if image is None:
        return None, [], f"请先上传{name}，然后在下方预览图上点击两点：上端在上，下端在下。"
    return (
        _draw_gravity_overlay(image, []),
        [],
        "可选：在下方预览图点击两点定义方向。上端在上，下端在下。",
    )


def _sync_source_gravity_preview(source_editor: dict[str, Any] | None):
    return _sync_gravity_preview(_source_background(source_editor), "源图")


def _sync_target_gravity_preview(target_editor: dict[str, Any] | None):
    return _sync_gravity_preview(_target_background(target_editor), "目标图")


def _reset_source_gravity_points(source_editor: dict[str, Any] | None):
    return _sync_source_gravity_preview(source_editor)


def _reset_target_gravity_points(target_editor: dict[str, Any] | None):
    return _sync_target_gravity_preview(target_editor)


def _update_gravity_points(
    image: np.ndarray | None,
    gravity_points: list[list[int]] | None,
    evt: gr.SelectData,
    image_name: str,
):
    if image is None:
        raise gr.Error(f"请先上传{image_name}。")

    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("未能识别点击坐标，请重试。")

    x, y = int(index[0]), int(index[1])
    points = [list(map(int, p)) for p in (gravity_points or [])]
    if len(points) >= 2:
        points = [[x, y]]
    else:
        points.append([x, y])

    preview = _draw_gravity_overlay(image, points)
    if len(points) == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        if dx == 0 and dy == 0:
            raise gr.Error("两点不能重合。")
        angle = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        status = f"已定义方向：{angle:.1f}°。保持上端在上，下端在下。"
        return preview, points, status, angle

    return preview, points, "已记录第 1 点，请再点击第 2 点。保持上端在上，下端在下。", gr.update()


def _update_source_gravity_points(
    source_editor: dict[str, Any] | None,
    source_gravity_points: list[list[int]] | None,
    evt: gr.SelectData,
):
    return _update_gravity_points(
        _source_background(source_editor),
        source_gravity_points,
        evt,
        "源图",
    )


def _update_target_gravity_points(
    target_editor: dict[str, Any] | None,
    target_gravity_points: list[list[int]] | None,
    evt: gr.SelectData,
):
    return _update_gravity_points(
        _target_background(target_editor),
        target_gravity_points,
        evt,
        "目标图",
    )


def _resolve_gravity_angle(
    gravity_mode: str,
    gravity_angle: float,
    gravity_points: list[list[int]] | None,
    name: str,
) -> float | None:
    if gravity_mode == "auto":
        return None
    if gravity_mode == "angle":
        return _normalize_angle_deg(gravity_angle)
    if gravity_mode == "two_points":
        points = gravity_points or []
        if len(points) != 2:
            raise gr.Error(f"当前{name}选择了“两点定义”，请先在预览上点击两点。")
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        if dx == 0 and dy == 0:
            raise gr.Error("两点不能重合。")
        return _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
    raise gr.Error(f"未知的{name}方向模式: {gravity_mode}")


def _validate_inputs(
    target_editor: dict[str, Any] | None,
    source_editor: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_img = _to_rgb((target_editor or {}).get("background"))
    source_img = _to_rgb((source_editor or {}).get("background"))

    if target_img is None:
        raise gr.Error("请先上传目标图。")
    if source_img is None:
        raise gr.Error("请先上传源图。")

    target_mask = _merge_editor_mask(target_editor)
    source_mask = _merge_editor_mask(source_editor)

    if target_mask is None or not np.any(target_mask):
        raise gr.Error("请在目标图上画出需要被填充的蒙版区域。")
    if source_mask is None or not np.any(source_mask):
        raise gr.Error("请在源图上画出用于填充的蒙版区域。")

    return target_img, target_mask, source_img, source_mask


def run_liquid_fill(
    target_editor: dict[str, Any] | None,
    source_editor: dict[str, Any] | None,
    target_gravity_mode: str,
    target_gravity_angle: float,
    target_gravity_points: list[list[int]] | None,
    source_gravity_mode: str,
    source_gravity_angle: float,
    source_gravity_points: list[list[int]] | None,
    mapping_mode: str,
    radial_method: str,
    gravity_strength: float,
    geometry_preservation: float,
    fill_ratio: float,
    num_boundary_points: int,
):
    target_img, target_mask, source_img, source_mask = _validate_inputs(
        target_editor, source_editor
    )
    resolved_target_gravity_angle = _resolve_gravity_angle(
        target_gravity_mode, target_gravity_angle, target_gravity_points, "目标图"
    )
    resolved_source_gravity_angle = _resolve_gravity_angle(
        source_gravity_mode, source_gravity_angle, source_gravity_points, "源图"
    )

    result, final_mask, info = liquid_fill_with_auto_gravity(
        target_mask=target_mask,
        source_img=source_img,
        source_mask=source_mask,
        target_gravity_angle=resolved_target_gravity_angle,
        source_gravity_angle=resolved_source_gravity_angle,
        mapping_mode=mapping_mode,
        radial_method=radial_method,
        gravity_strength=gravity_strength,
        geometry_preservation=geometry_preservation,
        fill_ratio=fill_ratio,
        num_boundary_points=num_boundary_points,
    )

    composite = target_img.copy()
    composite[final_mask > 0] = result[final_mask > 0]

    target_preview = _make_mask_preview(target_img, target_mask)
    source_preview = _make_mask_preview(source_img, source_mask)
    final_mask_preview = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)

    return result, composite, target_preview, source_preview, final_mask_preview, info


def build_demo() -> gr.Blocks:
    brush = gr.Brush(
        default_size=24,
        colors=["#ffffff"],
        default_color="#ffffff",
        color_mode="fixed",
    )

    with gr.Blocks(title="Liquid Fill UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Liquid Fill Mask UI
            左边上传需要被填充的目标图并画目标蒙版，右边上传内容来源图并画源蒙版。
            可以为源图额外指定一个手动重力方向，再去匹配目标蒙版的重力方向。
            """
        )

        target_gravity_points = gr.State([])
        source_gravity_points = gr.State([])

        with gr.Row():
            target_editor = gr.ImageEditor(
                label="目标图：画出需要被填充的区域",
                type="numpy",
                image_mode="RGBA",
                sources=["upload"],
                brush=brush,
                eraser=gr.Eraser(default_size=24),
                transforms=(),
                layers=True,
                canvas_size=(700, 520),
            )
            source_editor = gr.ImageEditor(
                label="源图：画出用于填充的区域",
                type="numpy",
                image_mode="RGBA",
                sources=["upload"],
                brush=brush,
                eraser=gr.Eraser(default_size=24),
                transforms=(),
                layers=True,
                canvas_size=(700, 520),
            )

        with gr.Row():
            target_gravity_mode = gr.Dropdown(
                choices=[
                    ("自动估计", "auto"),
                    ("手动角度", "angle"),
                    ("两点定义（上端在上，下端在下）", "two_points"),
                ],
                value="auto",
                label="目标图方向输入方式",
            )
            target_gravity_angle = gr.Slider(
                minimum=-180.0,
                maximum=180.0,
                value=90.0,
                step=1.0,
                label="目标图方向角度（向右为 0°，向下为 90°）",
            )
            clear_target_gravity_points = gr.Button("清空目标图两点方向")

        with gr.Row():
            target_gravity_preview = gr.Image(
                label="目标图方向预览：点击两点定义，保持上端在上、下端在下",
                type="numpy",
                sources=None,
                interactive=True,
            )
            target_gravity_status = gr.Textbox(
                label="目标图方向状态",
                value="请先上传目标图，然后在预览图上点击两点：上端在上，下端在下。",
                interactive=False,
            )

        with gr.Row():
            source_gravity_mode = gr.Dropdown(
                choices=[
                    ("自动估计", "auto"),
                    ("手动角度", "angle"),
                    ("两点定义（上端在上，下端在下）", "two_points"),
                ],
                value="auto",
                label="源图重力方向输入方式",
            )
            source_gravity_angle = gr.Slider(
                minimum=-180.0,
                maximum=180.0,
                value=90.0,
                step=1.0,
                label="源图重力方向角度（向右为 0°，向下为 90°）",
            )
            clear_source_gravity_points = gr.Button("清空两点方向")

        with gr.Row():
            source_gravity_preview = gr.Image(
                label="源图方向预览：点击两点定义，保持上端在上、下端在下",
                type="numpy",
                sources=None,
                interactive=True,
            )
            source_gravity_status = gr.Textbox(
                label="源图重力状态",
                value="请先上传源图，然后在预览图上点击两点：上端在上，下端在下。",
                interactive=False,
            )

        with gr.Row():
            mapping_mode = gr.Dropdown(
                choices=[
                    ("单调拉伸（内容保留优先）", "stretch"),
                    ("调和映射（贴轮廓优先）", "harmonic"),
                ],
                value="stretch",
                label="变形模式",
            )
            radial_method = gr.Dropdown(
                choices=["farthest", "pca", "furthest_from_center"],
                value="farthest",
                label="主轴检测方法",
            )
            gravity_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="重力强度",
            )
            geometry_preservation = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.75,
                step=0.05,
                label="几何保真强度",
            )
            fill_ratio = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="填充比例",
            )
            num_boundary_points = gr.Slider(
                minimum=80,
                maximum=500,
                value=300,
                step=10,
                label="边界采样点数",
            )

        run_button = gr.Button("运行填充", variant="primary")

        with gr.Row():
            result_image = gr.Image(label="算法输出", type="numpy")
            composite_image = gr.Image(label="叠加到目标图后的预览", type="numpy")

        with gr.Row():
            target_mask_preview = gr.Image(label="目标蒙版预览", type="numpy")
            source_mask_preview = gr.Image(label="源蒙版预览", type="numpy")
            final_mask_preview = gr.Image(label="最终输出蒙版", type="numpy")

        info_json = gr.JSON(label="调试信息")

        run_button.click(
            fn=run_liquid_fill,
            inputs=[
                target_editor,
                source_editor,
                target_gravity_mode,
                target_gravity_angle,
                target_gravity_points,
                source_gravity_mode,
                source_gravity_angle,
                source_gravity_points,
                mapping_mode,
                radial_method,
                gravity_strength,
                geometry_preservation,
                fill_ratio,
                num_boundary_points,
            ],
            outputs=[
                result_image,
                composite_image,
                target_mask_preview,
                source_mask_preview,
                final_mask_preview,
                info_json,
            ],
        )

        target_editor.change(
            fn=_sync_target_gravity_preview,
            inputs=[target_editor],
            outputs=[
                target_gravity_preview,
                target_gravity_points,
                target_gravity_status,
            ],
        )
        clear_target_gravity_points.click(
            fn=_reset_target_gravity_points,
            inputs=[target_editor],
            outputs=[
                target_gravity_preview,
                target_gravity_points,
                target_gravity_status,
            ],
        )
        target_gravity_preview.select(
            fn=_update_target_gravity_points,
            inputs=[target_editor, target_gravity_points],
            outputs=[
                target_gravity_preview,
                target_gravity_points,
                target_gravity_status,
                target_gravity_angle,
            ],
        )
        source_editor.change(
            fn=_sync_source_gravity_preview,
            inputs=[source_editor],
            outputs=[
                source_gravity_preview,
                source_gravity_points,
                source_gravity_status,
            ],
        )
        clear_source_gravity_points.click(
            fn=_reset_source_gravity_points,
            inputs=[source_editor],
            outputs=[
                source_gravity_preview,
                source_gravity_points,
                source_gravity_status,
            ],
        )
        source_gravity_preview.select(
            fn=_update_source_gravity_points,
            inputs=[source_editor, source_gravity_points],
            outputs=[
                source_gravity_preview,
                source_gravity_points,
                source_gravity_status,
                source_gravity_angle,
            ],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
