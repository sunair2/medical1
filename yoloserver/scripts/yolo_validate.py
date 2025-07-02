from pathlib import Path
import sys
import argparse
import logging

# 设置路径
current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'

if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

# 导入模块
from utils.data_validation import (
    verify_dataset_config,
    verify_split_uniqueness,
    delete_invalid_files
)
from logging_utils import setup_logger
from paths import LOGS_DIR, CONFIGS_DIR

if __name__ == '__main__':
    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="validate",
        model_name=None,
        log_level=logging.INFO,
        logger_name="yolo_validate"
    )

    parser = argparse.ArgumentParser(description="YOLO 数据集验证工具")
    parser.add_argument("--mode", "-m", type=str, default="SAMPLE",
                        choices=['SAMPLE', 'FULL'],
                        help="验证模式，SAMPLE表示只验证样本，FULL表示完整验证")
    parser.add_argument("--task", "-t", type=str, default="detection",
                        choices=["detection", "segmentation"],
                        help="任务类型，detection表示检测任务，segmentation表示分割任务")
    parser.add_argument("--delete-invalid", "-d", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="是否删除无效文件（支持 --delete-invalid / --no-delete-invalid）")
    args = parser.parse_args()

    VERIFY_MODE = args.mode
    TASK_TYPE = args.task
    ENABLE_DELETE_INVALID = args.delete_invalid

    logger.info(f"开始验证数据集，模式为: {VERIFY_MODE}，任务类型为: {TASK_TYPE}，删除非法数据: {ENABLE_DELETE_INVALID}")

    yaml_path = CONFIGS_DIR / "data.yaml"

    # 执行基础验证
    basic_validation_passed, invalid_data_list = verify_dataset_config(
        yaml_path, logger, mode=VERIFY_MODE, task_type=TASK_TYPE
    )

    basic_validation_handled = basic_validation_passed

    if not basic_validation_passed:
        logger.error("基础数据集验证未通过，请查看详细的日志")
        logger.error(f"检测到 {len(invalid_data_list)} 个不合法的数据样本，详细信息如下:")
        for i, item in enumerate(invalid_data_list):
            logger.error(f"不合法数据样本 {i + 1}：图像 {item['image_path']}，标签 {item['label_path']}，错误信息: {item['error_message']}")

        if ENABLE_DELETE_INVALID:
            if sys.stdin.isatty():
                print("是否删除这些非法数据样本？")
                print("请注意：删除操作不可逆")
                print("1. 是，删除图像和对应的标签文件")
                print("2. 否，保留文件")
                user_choice = input("请输入您的选择 (1 或 2): ")
                if user_choice == "1":
                    delete_invalid_files(invalid_data_list, logger)
                    basic_validation_handled = True
                    logger.info("已删除非法数据样本，基础验证问题已尝试处理")
                elif user_choice == "2":
                    logger.info("用户选择保留非法数据样本")
                    basic_validation_handled = False
                else:
                    logger.warning("输入错误，已取消删除非法数据样本")
            else:
                logger.warning("当前非交互式终端，将直接删除非法数据样本")
                delete_invalid_files(invalid_data_list, logger)
                basic_validation_handled = True
        else:
            logger.warning("检测到了不合法数据，但是未启用删除功能，文件将被保留")
            basic_validation_handled = False
    else:
        logger.info("基础数据集验证完成：通过")

    # 执行唯一性验证
    logger.info("开始执行分割唯一性验证")
    unique_validation_passed = verify_split_uniqueness(yaml_path, logger)

    if unique_validation_passed:
        logger.info("分割唯一性验证完成：通过")
    else:
        logger.error("分割唯一性验证未通过，存在重复文件，请查看详细的日志")

    if basic_validation_handled and unique_validation_passed:
        logger.info("数据集验证完成：通过")
    else:
        logger.error("数据集验证未通过，请查看详细的日志")


