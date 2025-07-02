# Safe/yoloserver/utils/infer_stream.py
from typing import Generator, Callable
import logging
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import argparse
from utils.infer_frame import process_frame
from utils import YOLOSERVER_ROOT

YOLO_SERVICE_DIR = YOLOSERVER_ROOT



def stream_inference(
    weights: str,
    source: str,
    project_args: dict,
    yolo_args: dict,
    pause_callback: Callable[[], bool] = lambda: False
) -> Generator[tuple[np.ndarray, np.ndarray, object], None, None]:
    logger = logging.getLogger("YOLO_Training")
    logger.info("===== YOLOv8 安全帽检测 UI 推理开始 =====")

    model = None
    cap = None
    video_writer = None
    last_frame = None
    last_annotated = None
    try:
        # 分辨率和美化参数（现有代码）
        resolution_map = {
            360: (640, 360),
            720: (1280, 720),
            1080: (1920, 1080),
            1440: (2560, 1440),
        }

        display_width, display_height = resolution_map[project_args.get('display_size', 720)]
        beautify_params = {
            "use_chinese_mapping": project_args.get('use_chinese_mapping', True),
            "beautify": project_args.get('beautify', True),
            "font_size": project_args.get('font_size', 22),
            "line_width": project_args.get('line_width', 4),
            "label_padding": (project_args.get('label_padding_x', 30), project_args.get('label_padding_y', 18)),
            "radius": project_args.get('radius', 8),
            "text_color": (0, 0, 0),
            "font_path": project_args.get('font_path', 'LXGWWenKai-Bold.ttf'),
        }

        # 路径验证
        model_path = Path(weights)
        if not model_path.is_absolute():
            model_path = YOLO_SERVICE_DIR / "models" / "checkpoints" / weights
        if not source.isdigit():
            source_path = Path(source)
            if not source_path.exists():
                logger.error(f"输入源不存在: {source_path}")
                raise FileNotFoundError(f"输入源不存在: {source_path}")
            source = str(source_path)
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {weights}")

        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(str(model_path))

        project_args_ns = argparse.Namespace(**project_args)
        yolo_args_dict = yolo_args.copy()
        yolo_args_dict['stream'] = True

        # 流式推理
        if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not cap.isOpened():
                logger.error(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
                raise RuntimeError(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            display_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            display_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"视频分辨率: {display_width}x{display_height}, 帧率: {fps}")
            frames_dir = None
            save_dir = None
            idx = 0
            while cap.isOpened():
                if pause_callback():
                    logger.debug("推理暂停")
                    if last_frame is not None:
                        yield last_frame, last_annotated, None  # 返回最后一帧
                    cv2.waitKey(1)
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, **yolo_args_dict)
                result = next(iter(results))
                if idx == 0:
                    save_dir = Path(result.save_dir)
                    logger.info(f"推理结果保存目录: {save_dir}")
                    if project_args.get('save_frame', False):
                        frames_dir = save_dir / "0_frames"
                        frames_dir.mkdir(parents=True, exist_ok=True)
                    if project_args.get('save', False):
                        video_path = save_dir / "output.mp4"
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (int(display_width), int(display_height))
                        )
                annotated_frame = process_frame(frame, result, project_args_ns, beautify_params)
                if video_writer:
                    video_writer.write(annotated_frame)
                if frames_dir:
                    frame_path = frames_dir / f"{idx}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                    logger.debug(f"保存帧图像: {frame_path}")

                last_frame, last_annotated = frame, annotated_frame
                yield frame, annotated_frame, result
                idx += 1
            logger.info(f"{'摄像头' if source.isdigit() else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")
        else:
            source_path = Path(source)
            image_files = [source_path] if source_path.is_file() else sorted(source_path.glob("*.[jp][pn][gf]"))
            if not image_files:
                logger.error("目录中无图片文件")
                raise ValueError("目录中无图片文件")
            save_dir = None
            for idx, img_path in enumerate(image_files):
                if pause_callback():
                    logger.debug("推理暂停")
                    if last_frame is not None:
                        yield last_frame, last_annotated, None
                    cv2.waitKey(1)
                    continue
                raw_frame = cv2.imread(str(img_path))
                if raw_frame is None:
                    logger.warning(f"无法读取图片: {img_path}")
                    continue
                results = model.predict(source=str(img_path), **yolo_args_dict)
                result = next(iter(results))
                if idx == 0:
                    save_dir = Path(result.save_dir)
                    logger.info(f"推理结果保存目录: {save_dir}")
                annotated_frame = process_frame(raw_frame, result, project_args_ns, beautify_params)
                if project_args.get('save', False):
                    output_path = save_dir / Path(result.path).name
                    cv2.imwrite(str(output_path), annotated_frame)
                    logger.debug(f"保存图像: {output_path}")
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
                raw_frame = cv2.resize(raw_frame, (display_width, display_height))
                last_frame, last_annotated = raw_frame, annotated_frame
                yield raw_frame, annotated_frame, result
            logger.info(f"推理完成，结果已保存至: {save_dir or '未保存'}")

        if save_dir and project_args.get('save', False):
            from . import rename_log_file
            rename_log_file(logger, save_dir, Path(weights).stem, project_args.get('log_encoding', 'utf-8-sig'))

    except Exception as e:
        logger.error(f"UI 推理失败: {e}", exc_info=True)
        raise
    finally:
        if video_writer:
            video_writer.release()
        if cap:
            cap.release()
        if model:
            model = None
        logger.info("===== YOLOv8 安全帽检测 UI 推理结束 =====")