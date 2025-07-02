# -*- coding:utf-8 -*-
"""
项目初始化脚本，检查并创建必要的项目结构，提示用户将原始数据存放到指定的位置
"""
import logging
from utils.paths import (
    YOLOSERVER_ROOT,
    CONFIGS_DIR, DATA_DIR, RUNS_DIR, LOGS_DIR, MODELS_DIR, PRETRAINED_DIR, CHECKPOINTS_DIR, SCRIPTS_DIR,
    RAW_IMAGES_DIR, ORIGINAL_ANNOTATIONS_DIR
)
from utils.logging_utils import setup_logger
from utils.performance_utils import time_it

def safe_relpath(path, root):
    try:
        return path.relative_to(root)
    except Exception:
        return path

# 第一步：配置日志记录
logger = setup_logger(
    base_path=LOGS_DIR,
    log_type="init_project",
    log_level=logging.INFO,
    logger_name="YOLO Initialize Project"
)

# 第二步：定义项目初始化函数
@time_it(iterations=1, name="项目初始化", logger_instance=logger)
def initialize_project():
    """
    检查并创建项目所需的文件夹结构
    :return:
    """
    logger.info("初始化项目结构开始".center(60, "═"))
    logger.info(f"当前项目的根目录为：{YOLOSERVER_ROOT.resolve()}")
    created_dirs = []
    existing_dirs = []
    raw_data_status = []
    standard_data_to_create = [
        CONFIGS_DIR, DATA_DIR, RUNS_DIR, MODELS_DIR, CHECKPOINTS_DIR, PRETRAINED_DIR, LOGS_DIR, SCRIPTS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "val" / "images",
        DATA_DIR / "test" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "labels",
    ]
    logger.info("检查并创建标准项目目录结构".center(60, "═"))
    for d in standard_data_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"已成功创建项目目录：{safe_relpath(d, YOLOSERVER_ROOT)}")
                created_dirs.append(safe_relpath(d, YOLOSERVER_ROOT))
            except Exception as e:
                logger.error(f"创建目录失败：{d} 失败：{e}")
                created_dirs.append(f"创建目录：{d} 失败：{e}")
        else:
            logger.info(f"检测到目录已存在：{safe_relpath(d, YOLOSERVER_ROOT)}")
            existing_dirs.append(safe_relpath(d, YOLOSERVER_ROOT))
    logger.info("标准数据集项目结构检查完成".center(60, "═"))
    # 第三步：检查原始数据集目录并给出提示
    logger.info("检测原始数据集目录并给出提示".center(60, "═"))
    raw_dirs_to_check = {
        "原始图像": RAW_IMAGES_DIR,
        "原始标注": ORIGINAL_ANNOTATIONS_DIR,
    }
    for desc, raw_dir in raw_dirs_to_check.items():
        if not raw_dir.exists():
            logger.warning(f"{desc} 目录尚未创建，请将原始数据或标注数据放置在如下目录：\n{raw_dir.resolve()}")
            raw_data_status.append(f"{desc} 目录不存在，请将原始数据放置于：{raw_dir.resolve()}")
        else:
            if not any(raw_dir.iterdir()):
                msg = f"{desc} 已经存在，但空目录。请将数据({desc})放在此目录下，以便完成建模过程准备。"
                logger.warning(msg)
                raw_data_status.append(f"{safe_relpath(raw_dir, YOLOSERVER_ROOT)} 已经存在，但为空目录，请尽早完成数据准备")
            else:
                logger.info(f"{desc} 已经存在，{safe_relpath(raw_dir, YOLOSERVER_ROOT)} 有内容存在")
                raw_data_status.append(f"{safe_relpath(raw_dir, YOLOSERVER_ROOT)} 已存在")

    # 第四步：汇总所有创建信息和数据集检测情况
    logger.info("汇总初始化过程生成的相关信息".center(60, "═"))
    if created_dirs:
        logger.info(f"一共创建了 {len(created_dirs)} 个目录：")
        for d in created_dirs:
            logger.info(f" - {d}")
    else:
        logger.info("本次初始化未创建任何目录")
    if existing_dirs:
        logger.info(f"检测到已有目录共 {len(existing_dirs)} 个 已经存在的目录有：")
        for d in existing_dirs:
            logger.info(f" - {d}")
    if raw_data_status:
        logger.info("原始数据目录检查结果如下".center(60, "═"))
        for s in raw_data_status:
            logger.info(f" - {s}")
        logger.info("原始数据目录检查结束".center(60, "═"))

if __name__ == "__main__":
    initialize_project()