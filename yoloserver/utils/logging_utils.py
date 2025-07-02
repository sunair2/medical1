import logging
from datetime import datetime
from pathlib import Path
import platform
import psutil
import torch
#1.构建日志目录完整的存放目录

#2.生成带时间戳的日志文件名

def setup_logger(base_path:Path, log_type:str = "general",
                 model_name:str = None,
                 encoding:str = "utf-8",
                 log_level : int = logging.INFO,
                 temp_log:bool = False,
                 logger_name:str = "default"
                 ):
    """

    :param base_path:       日志记文件的根路径
    :param log_type:        日志的类型
    :param model_name:     模型名称
    :param encoding:        编码格式
    :param log_level:       日志等级
    :param temp_log:        是否使用临时命名
    :param logger_name:     日志记录器的名称
    :return:                返回一个日志记录器
    """

    # 1. 构建日志文件完整的存放路径
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成一个带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #根据temp_log参数生成不同的日志名
    prefix = "temp" if temp_log else log_type.replace(" ","-")
    log_filename_parts = [prefix, timestamp]
    if model_name:
            log_filename_parts.append(model_name.replace(" ", "-"))
    log_filename = "_".join(log_filename_parts)+".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定名称的logger实例
    logger = logging.getLogger(logger_name)
    # 设置日志记录器记录最低记录级别
    logger.setLevel(log_level)
    #组织日志事件传播到父级目录logger
    logger.propagate = False

    # 4. 需要避免重复添加日志处理器，因此先检查日志处理器列表中是否已经存在了指定的日志处理器
    if logger.hasHandlers():
            for handler in logger.handlers:
                logger.removeHandler(handler)

    # 5.创建文件处理器，将日志写入到文件当中
    file_handler = logging.FileHandler(log_file,encoding=encoding)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # 将文件处理器添加到logger实例中
    logger.addHandler(file_handler)

    # 6. 创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    # 将控制台处理器添加到logger实例当中
    logger.addHandler(console_handler)

    #输出一些初始化信息到日志，确认配置成功

    logger.info(f"日志记录器已启动，日志文件保存路径为:{log_file}")
    logger.info(f"日志记录器的根目录：              {base_path}")
    logger.info(f"日志记录器的名称:                {logger_name}")
    logger.info(f"日志记录器的类型:                {log_type}")
    logger.info(f"日志记录器的级别：               {logging.getLevelName(log_level)}")

    logger.info(f"相关设备".center(60,"="))
    logger.info(f"操作系统: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"处理器: {platform.processor()}")
    logger.info(f"CPU核心数: {psutil.cpu_count(logical=True)}")
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU占用率: {cpu_percent}%")
    mem = psutil.virtual_memory()
    logger.info(f"内存占用: {round(mem.used / (1024 ** 3), 2)} GB / {round(mem.total / (1024 ** 3),2)} GB ({mem.percent}%)")

    logger.info(f"GPU".center(60,"="))

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    gpu_available = torch.cuda.is_available()
    logger.info(f"PyTorch版本: {torch_version}")
    logger.info(f"CUDA版本: {cuda_version}")
    logger.info(f"cuDNN版本: {cudnn_version}")
    logger.info(f"GPU可用: {'是' if gpu_available else '否'}")
    if gpu_available:
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        mem_allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2)
        mem_reserved = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 2)
        total_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        logger.info(f"已分配显存: {mem_allocated} GB, 已保留显存: {mem_reserved} GB, 总显存: {total_memory} GB")

    logger.info(f"日志记录器初始化成功".center(60, "="))

    return logger
def rename_log_file(logger_obj, save_dir, model_name, encoding="utf-8"):
    """
    重命名日志文件，如 train1_20250626_modelA.log

    :param logger_obj: 日志记录器对象
    :param save_dir: 保存日志的目录路径
    :param model_name: 模型名称，用于命名日志文件
    :param encoding: 文件编码，默认 utf-8
    :return: None
    """
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            timestamp_parts = old_log_file.stem.split("_")
            if len(timestamp_parts) < 3:
                logger_obj.warning(f"日志文件名无法解析时间戳: {old_log_file.name}")
                continue
            timestamp = timestamp_parts[2]

            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp}_{model_name}.log"

            # 关闭旧的日志处理器
            handler.close()
            logger_obj.removeHandler(handler)

            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger_obj.info(f"日志文件已经重命名成功: {new_log_file}")
                except OSError as e:
                    logger_obj.error(f"日志文件重命名失败: {e}")
                    # 尝试重新添加旧处理器
                    re_added_handler = logging.FileHandler(old_log_file, encoding=encoding)
                    re_added_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                    logger_obj.addHandler(re_added_handler)
                    continue
            else:
                logger_obj.warning(f"尝试重命名的日志文件不存在: {old_log_file}")
                continue

            # 重命名成功后添加新的日志处理器
            new_handler = logging.FileHandler(new_log_file, encoding=encoding)
            new_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger_obj.addHandler(new_handler)
            break


def log_parameters(args, exclude_params=None, logger=None):
    """
    记录命令行和 YAML 参数信息，返回结构化字典。

    Args:
        args: 命令行参数 (Namespace 对象)
        exclude_params: 不记录的参数键列表
        logger: 日志记录器实例

    Returns:
        dict: 参数字典，包含参数值和来源（命令行或 YAML）
    """
    if logger is None:
        logger = logging.getLogger("YoLo_Training")

    if exclude_params is None:
        exclude_params = ['log_encoding', 'log_level', 'extra_args', 'use_yaml']

    logger.info("开始模型参数信息".center(40, '='))
    logger.info("Parameters")
    logger.info("_" * 40)

    params_dict = {}

    for key, value in vars(args).items():
        if key not in exclude_params and not key.endswith('_specified'):
            source = '命令行' if getattr(args, f'{key}_specified', False) else 'YAML'
            logger.info(f"{key:<20}: {value} (来源: [{source}])")
            params_dict[key] = {
                'value': value,
                'source': source
            }

    return params_dict
