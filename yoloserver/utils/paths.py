#定义所有的路径信息

from pathlib import Path

#项目的根目录
YOLOSERVER_ROOT = Path(__file__).resolve().parents[1]

#配置文件目录
CONFIGS_DIR = YOLOSERVER_ROOT / "configs"

#模型目录
MODELS_DIR = YOLOSERVER_ROOT / "models"

#训练好的模型存放目录
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

#预训练模型存放目录
PRETRAINED_DIR = MODELS_DIR/"pretrained"

#模型的运行结果存放目录
RUNS_DIR = YOLOSERVER_ROOT / "runs"

#数据文件目录
DATA_DIR = YOLOSERVER_ROOT / "data"

#原始数据文件存放目录
RAW_DATA_DIR = DATA_DIR / "raw"

#原始图像文件存放目录
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"

#原始非yolo格式的数据
ORIGINAL_ANNOTATIONS_DIR = RAW_DATA_DIR / "original_annotations"

#yolo格式标注文件暂时存放目录
YOLO_STAGED_LABELS_DIR = RAW_DATA_DIR / "yolo_staged_labels"

#临时文件存放目录
RAW_TEMP_DIR = RAW_DATA_DIR / "temp"

#训练验证测试数据集存放目录
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"

#日志目录
LOGS_DIR = YOLOSERVER_ROOT / "logs"

#训练推理脚本存放目录
SCRIPTS_DIR = YOLOSERVER_ROOT / "scripts"

