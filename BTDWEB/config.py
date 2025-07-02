# -*- coding: utf-8 -*-
"""
BTDWEB 配置文件
包含API密钥、模型配置等敏感信息
"""

# DeepSeek API 配置
API_CONFIG = {
    "url": "https://api.siliconflow.cn/v1/chat/completions",
    "api_key": "sk-frtbnruyvpvlakdkkgzxiqkwhupjkiwxiybpzjoimnlmblpz",
    "model": "Qwen/QwQ-32B",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 30
}

# Flask 应用配置
FLASK_CONFIG = {
    "debug": True,
    "send_file_max_age_default": 0
}

# 文件上传配置
UPLOAD_CONFIG = {
    "upload_folder": "static/uploads",
    "detect_folder": "static/detections", 
    "models_folder": "static/models",
    "max_file_size": 16 * 1024 * 1024,  # 16MB
    "allowed_image_extensions": {"png", "jpg", "jpeg", "gif", "bmp", "tiff"},
    "allowed_model_extensions": {"pt"}
}

# AI分析提示词模板
ANALYSIS_PROMPT_TEMPLATE = """
作为一名专业的医学影像AI分析专家，请分析以下YOLO肿瘤检测结果：

检测文件：{filename}
检测结果：
{detection_results}

请从以下几个方面进行专业分析，并以纯文本格式（不要使用Markdown）输出：

1. 检测结果总结：简要概述检测到的对象和置信度
2. 医学意义：如果检测到肿瘤相关对象，请分析其可能的医学意义
3. 风险评估：基于检测结果给出初步的风险评估
4. 建议措施：给出后续的医学建议或需要注意的事项
5. 技术评价：评价检测模型的表现和置信度

请用中文回答，语言要专业但易懂，适合医生和患者理解。请使用纯文本格式，不要使用Markdown格式的标题、列表和其他格式化元素。
注意：这只是AI辅助分析，不能替代专业医生的诊断，请务必咨询专业医生。
"""

SYSTEM_PROMPT = "你是一名专业的医学影像AI分析专家，擅长解读YOLO模型的肿瘤检测结果，能够提供专业的医学分析和建议。你的分析应该客观、准确，并以纯文本格式（不使用Markdown标记）输出，强调需要专业医生的进一步诊断。"
