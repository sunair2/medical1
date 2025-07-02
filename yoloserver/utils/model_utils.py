from datetime import  datetime
from pathlib import Path
import shutil
import os
import logging

def copy_checkpoint_models(train_dir,model_filename, checkpoint_dir,logger):
    """
    复制模型到指定的地点
    :param train_dir:
    :param model_filename:
    :param checkpoint_dir:
    :param logger:
    :return:
    """
    if not isinstance(train_dir, Path) or not train_dir.is_dir():
        logger.error(f"{train_dir} 是一个无效的路径")
        return

    if not isinstance(checkpoint_dir, Path) or not checkpoint_dir.is_dir():
        logger.error(f"{checkpoint_dir} 是一个无效的路径,不能存储训练好的模型")
        return

    # 准备准备新的模型文件名
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_name = Path(model_filename).stem
    train_suffix = train_dir.name

    # 遍历进行复制
    for model_type in ["best", "last"]:
        src_path = train_dir / "weights" / f"{model_type}.pt"
        if src_path.exists():
            checkpoint_name = f"{train_suffix}_{date_str}_{base_model_name}_{model_type}.pt"
            dest_path = checkpoint_dir / checkpoint_name
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"{model_type}模型已经从{src_path}复制到至{dest_path}")
            except FileNotFoundError:
                logger.warning(f"{model_type}模型不存在")
            except shutil.SameFileError:
                logger.error(f"源文件和目标文件相同,无法复制")
            except PermissionError:
                logger.error(f"没有权限复制文件")
            except OSError as e:
                logger.error(f"复制文件时出错: {e}")
        else:
            logger.warning(f"{model_type}.pt 模型不存在,不存在预期的源路径")

            import logging

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
