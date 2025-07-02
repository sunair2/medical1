import os
from flask import Flask, request, render_template, send_file, session, jsonify
import sys
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime
import requests
import json
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from config import API_CONFIG, FLASK_CONFIG, UPLOAD_CONFIG, ANALYSIS_PROMPT_TEMPLATE, SYSTEM_PROMPT

app = Flask(__name__)

# 从配置文件加载配置
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = FLASK_CONFIG['send_file_max_age_default']
app.secret_key = 'your-secret-key-here'  # 用于会话管理

# 设置上传与检测目录
UPLOAD_FOLDER = UPLOAD_CONFIG['upload_folder']
DETECT_FOLDER = UPLOAD_CONFIG['detect_folder']
MODELS_FOLDER = UPLOAD_CONFIG['models_folder']
LOGS_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')
REPORTS_PDF_FOLDER = os.path.join(os.path.dirname(__file__), 'reports_pdf')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_PDF_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER


def validate_file(file, file_type='image'):
    """
    验证上传的文件
    """
    if not file or file.filename == '':
        return False, "未选择文件"

    if file_type == 'image':
        allowed_extensions = UPLOAD_CONFIG['allowed_image_extensions']
        error_msg = "不支持的图像格式，请上传 PNG、JPG、JPEG 等格式的图片"
    else:  # model
        allowed_extensions = UPLOAD_CONFIG['allowed_model_extensions']
        error_msg = "不支持的模型格式，请上传 .pt 格式的模型文件"

    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_extension not in allowed_extensions:
        return False, error_msg

    return True, "文件验证通过"


def analyze_detection_results(detections, image_filename):
    """
    使用AI API分析检测结果
    """
    try:
        # 构建分析提示词
        detection_text = "\n".join(detections) if detections else "未检测到任何对象"

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            filename=image_filename,
            detection_results=detection_text
        )

        payload = {
            "model": API_CONFIG['model'],
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": API_CONFIG['temperature'],
            "max_tokens": API_CONFIG['max_tokens']
        }

        headers = {
            "Authorization": f"Bearer {API_CONFIG['api_key']}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            API_CONFIG['url'],
            json=payload,
            headers=headers,
            timeout=API_CONFIG['timeout']
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis = result['choices'][0]['message']['content']
                return analysis
            else:
                return "API响应格式异常，无法获取分析结果。"
        else:
            return f"API请求失败，状态码：{response.status_code}，错误信息：{response.text}"

    except requests.exceptions.Timeout:
        return "AI分析请求超时，请稍后重试。"
    except requests.exceptions.RequestException as e:
        return f"AI分析请求失败：{str(e)}"
    except Exception as e:
        return f"AI分析过程中发生错误：{str(e)}"


def generate_pdf_report(detections, ai_analysis, image_filename, model_name, original_image_path=None, processed_image_path=None, doctor_name=None, high_confidence_warnings=None):
    """
    生成PDF报告，支持中文字体和图片展示
    """
    try:
        # 注册中文字体
        font_path = os.path.join(os.path.dirname(__file__), 'LXGWWenKai-Bold.ttf')
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            chinese_font = 'ChineseFont'
        else:
            # 如果找不到中文字体，使用系统默认字体
            chinese_font = 'Helvetica'
        
        # 生成报告编号和时间戳
        timestamp = datetime.now()
        time_str = timestamp.strftime("%Y/%m/%d_%H:%M:%S")
        report_id = timestamp.strftime("%Y%m%d%H%M%S")
        
        # 生成PDF文件名：模型名-时间-编号
        pdf_filename = f"{model_name}-{timestamp.strftime('%Y%m%d_%H%M%S')}-{report_id}.pdf"
        pdf_path = os.path.join(REPORTS_PDF_FOLDER, pdf_filename)
        
        # 创建PDF文档
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # 获取样式
        styles = getSampleStyleSheet()
        
        # 定义自定义样式（使用中文字体）
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.blue,
            spaceAfter=30,
            fontName=chinese_font
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            fontName=chinese_font
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            fontName=chinese_font
        )
        
        warning_style = ParagraphStyle(
            'WarningStyle',
            parent=styles['Normal'],
            fontSize=11,
            fontName=chinese_font,
            textColor=colors.red,
            borderColor=colors.red,
            borderWidth=1,
            borderPadding=5,
            backColor=colors.pink,
            borderRadius=5
        )
        
        # 添加标题
        title = Paragraph("YOLO肿瘤检测与AI分析报告", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # 添加基本信息表格
        basic_info_data = [
            ['报告编号', report_id],
            ['生成时间', timestamp.strftime("%Y/%m/%d %H:%M:%S")],
            ['检测模型', model_name],
            ['分析图像', image_filename],
        ]
        
        # 添加医生署名信息
        if doctor_name:
            basic_info_data.append(['医生署名', doctor_name])
        
        basic_info_table = Table(basic_info_data, colWidths=[2*inch, 4*inch])
        basic_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(basic_info_table)
        story.append(Spacer(1, 30))
        
        # 添加高置信度警告（如果有）
        if high_confidence_warnings and len(high_confidence_warnings) > 0:
            warning_heading = Paragraph("⚠️ 高置信度警告", ParagraphStyle(
                'WarningHeading',
                parent=heading_style,
                textColor=colors.red
            ))
            story.append(warning_heading)
            
            for warning in high_confidence_warnings:
                warning_para = Paragraph(warning, warning_style)
                story.append(warning_para)
                story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
        
        # 添加检测结果
        detection_heading = Paragraph("检测结果详情", heading_style)
        story.append(detection_heading)
        
        if detections and detections[0] != "未检测到任何对象":
            detection_data = [['检测对象', '置信度']]
            for detection in detections:
                if ':' in detection:
                    obj, conf = detection.split(':', 1)
                    detection_data.append([obj.strip(), conf.strip()])
            
            detection_table = Table(detection_data, colWidths=[3*inch, 2*inch])
            detection_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(detection_table)
        else:
            no_detection = Paragraph("未检测到任何肿瘤对象", normal_style)
            story.append(no_detection)
        
        story.append(Spacer(1, 30))
        
        # 添加AI分析结果
        ai_heading = Paragraph("AI智能分析报告", heading_style)
        story.append(ai_heading)
        
        # 处理AI分析文本，确保正确显示
        if ai_analysis:
            # 将长文本分段
            ai_paragraphs = ai_analysis.split('\n')
            for para in ai_paragraphs:
                if para.strip():
                    ai_para = Paragraph(para.strip(), normal_style)
                    story.append(ai_para)
                    story.append(Spacer(1, 6))
        else:
            no_analysis = Paragraph("AI分析不可用", normal_style)
            story.append(no_analysis)
        
        story.append(Spacer(1, 30))
        
        # 添加图片对比部分
        if original_image_path and processed_image_path:
            if os.path.exists(original_image_path) and os.path.exists(processed_image_path):
                image_heading = Paragraph("处理前后图像对比", heading_style)
                story.append(image_heading)
                story.append(Spacer(1, 15))
                
                # 创建图片对比表格
                try:
                    # 调整图片大小以适应PDF页面
                    img_width = 2.5 * inch
                    img_height = 2 * inch
                    
                    # 创建原始图片
                    original_img = ReportLabImage(original_image_path, width=img_width, height=img_height)
                    processed_img = ReportLabImage(processed_image_path, width=img_width, height=img_height)
                    
                    # 创建标题
                    original_title = Paragraph("处理前图像", normal_style)
                    processed_title = Paragraph("处理后图像", normal_style)
                    
                    # 创建图片对比表格
                    image_data = [
                        [original_title, processed_title],
                        [original_img, processed_img]
                    ]
                    
                    image_table = Table(image_data, colWidths=[3*inch, 3*inch])
                    image_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('FONTNAME', (0, 0), (-1, 0), chinese_font),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
                    ]))
                    
                    story.append(image_table)
                    story.append(Spacer(1, 20))
                    
                except Exception as img_error:
                    print(f"添加图片失败: {str(img_error)}")
                    no_images = Paragraph("图片加载失败，无法显示对比图像", normal_style)
                    story.append(no_images)
                    story.append(Spacer(1, 15))
            else:
                no_images = Paragraph("原始图片或处理后图片文件不存在", normal_style)
                story.append(no_images)
                story.append(Spacer(1, 15))
        
        # 构建PDF
        doc.build(story)
        
        return pdf_path, pdf_filename
        
    except Exception as e:
        print(f"PDF生成失败: {str(e)}")
        return None, None


@app.route('/', methods=['GET', 'POST'])
def upload_detect():
    if request.method == 'POST':
        # 获取客户端上传的图片和模型
        image_file = request.files.get("image")
        model_file = request.files.get("model")
        doctor_name = request.form.get("doctor_name")
        perform_ai_analysis = 'perform_ai_analysis' in request.form

        # 验证医生姓名
        if not doctor_name or doctor_name.strip() == '':
            return render_template(
                'index.html',
                prediction="请输入医生姓名，此信息必填",
                detections=[],
                ai_analysis="",
                image_path=None
            )

        # 验证文件
        if not image_file or not model_file:
            return render_template(
                'index.html',
                prediction="请同时选择图片文件和模型文件",
                detections=[],
                ai_analysis="",
                image_path=None
            )

        # 验证图片文件
        img_valid, img_msg = validate_file(image_file, 'image')
        if not img_valid:
            return render_template(
                'index.html',
                prediction=f"图片文件错误：{img_msg}",
                detections=[],
                ai_analysis="",
                image_path=None
            )

        # 验证模型文件
        model_valid, model_msg = validate_file(model_file, 'model')
        if not model_valid:
            return render_template(
                'index.html',
                prediction=f"模型文件错误：{model_msg}",
                detections=[],
                ai_analysis="",
                image_path=None
            )

        # 处理上传的图片
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        img_filename = f"{timestamp}_{image_file.filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], img_filename)
        detect_path = os.path.join(app.config["DETECT_FOLDER"], img_filename)
        image_file.save(upload_path)

        # 处理上传的模型
        model_filename = f"{timestamp}_{model_file.filename}"
        model_path = os.path.join(app.config["MODELS_FOLDER"], model_filename)
        model_file.save(model_path)

        try:
            # 使用上传的模型进行目标检测
            model = YOLO(model_path)

            # 进行目标检测
            results = model(upload_path)

            # 绘制检测结果图像并保存
            result_img_array = results[0].plot()
            result_pil = Image.fromarray(result_img_array)
            result_pil.save(detect_path)

            # 提取检测框信息（标签 + 置信度）
            detections = []
            high_confidence_warnings = []  # 存储高置信度警告
            boxes = results[0].boxes
            if boxes is not None and boxes.cls.numel() > 0:
                for cls_id, conf in zip(boxes.cls, boxes.conf):
                    class_name = model.names[int(cls_id)]
                    confidence = round(float(conf) * 100, 2)
                    detection_text = f"{class_name}: {confidence}%"
                    detections.append(detection_text)
                    
                    # 添加高置信度警告 (>90%)
                    if confidence > 90:
                        warning_text = f"警告：检测到置信度高达{confidence}%的{class_name}，建议立即进行进一步诊断与干预！"
                        high_confidence_warnings.append(warning_text)
            else:
                detections.append("未检测到任何对象")

            # 使用AI分析检测结果（如果用户选择了进行AI分析）
            ai_analysis = ""
            if perform_ai_analysis:
                print("正在进行AI分析...")  # 调试信息
                ai_analysis = analyze_detection_results(detections, img_filename)
                print(f"AI分析完成: {ai_analysis[:100]}...")  # 调试信息
            else:
                ai_analysis = "用户选择不进行AI智能分析"

            # 清理临时文件（可选）
            try:
                os.remove(model_path)  # 删除临时模型文件
            except:
                pass

            # 自动生成PDF文件，不等待用户点击下载
            model_name = model_file.filename.rsplit('.', 1)[0] if '.' in model_file.filename else model_file.filename
            original_image_path = os.path.join(app.config["UPLOAD_FOLDER"], img_filename)
            processed_image_path = os.path.join(app.config["DETECT_FOLDER"], img_filename)
            
            # 生成PDF
            pdf_path, pdf_filename = generate_pdf_report(
                detections, 
                ai_analysis, 
                img_filename, 
                model_name,
                original_image_path,
                processed_image_path,
                doctor_name,
                high_confidence_warnings=high_confidence_warnings
            )

            # 保存分析结果到session，用于PDF下载
            session['last_analysis'] = {
                'detections': detections,
                'ai_analysis': ai_analysis,
                'image_filename': img_filename,
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'doctor_name': doctor_name,
                'high_confidence_warnings': high_confidence_warnings,  # 添加高置信度警告
                'pdf_filename': pdf_filename if pdf_path else None  # 添加PDF文件名
            }
            
            # 立即记录到日志文件，包含PDF信息
            try:
                # 生成报告编号和时间戳
                now_timestamp = datetime.now()
                report_id = now_timestamp.strftime("%Y%m%d%H%M%S")
                
                # 准备日志条目
                log_entry = {
                    'timestamp': now_timestamp.isoformat(),
                    'report_id': report_id,
                    'image_filename': img_filename,
                    'model_name': model_name,
                    'detections_count': len(detections) if detections and detections[0] != "未检测到任何对象" else 0,
                    'detections': detections,
                    'ai_analysis': ai_analysis,
                    'doctor_name': doctor_name,
                    'high_confidence_warnings': high_confidence_warnings if high_confidence_warnings else [],
                    'pdf_filename': pdf_filename if pdf_path else None  # 添加PDF文件名
                }
                
                # 添加图片路径信息
                log_entry['image_path'] = f"detections/{img_filename}"
                
                # 读取现有日志或创建新的
                log_filename = f"report_log_{now_timestamp.strftime('%Y%m%d')}.json"
                log_path = os.path.join(LOGS_FOLDER, log_filename)
                
                logs = []
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            logs = json.load(f)
                    except:
                        logs = []
                
                logs.append(log_entry)
                
                # 写入日志
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                    
                print(f"检测记录已保存到日志: {report_id}")
            except Exception as log_error:
                print(f"保存检测记录到日志失败: {str(log_error)}")

            return render_template(
                'index.html',
                prediction="检测和分析完成",
                detections=detections,
                ai_analysis=ai_analysis,
                image_path=f"detections/{img_filename}",
                show_download_btn=True if pdf_path else False,
                high_confidence_warnings=high_confidence_warnings,  # 传递高置信度警告到模板
                pdf_filename=pdf_filename if pdf_path else None  # 传递PDF文件名
            )

        except Exception as e:
            # 清理可能的临时文件
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
            except:
                pass

            return render_template(
                'index.html',
                prediction=f"检测过程中发生错误：{str(e)}",
                detections=[],
                ai_analysis="由于检测失败，无法进行AI分析。",
                image_path=None
            )

    return render_template('index.html', prediction=None)


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    """
    下载PDF报告
    """
    try:
        data = request.get_json()
        pdf_filename = data.get('pdf_filename', '')
        
        if not pdf_filename:
            # 如果没有传入PDF文件名，尝试从session获取
            if 'last_analysis' in session and 'pdf_filename' in session['last_analysis']:
                pdf_filename = session['last_analysis']['pdf_filename']
            else:
                return jsonify({'error': '无法获取PDF文件名'}), 400
        
        # 构建PDF文件路径
        pdf_path = os.path.join(REPORTS_PDF_FOLDER, pdf_filename)
        
        # 检查PDF文件是否存在
        if pdf_path and os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'PDF文件不存在或已被删除'}), 404
            
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500


@app.route('/history')
def view_history():
    """
    查看检测历史记录
    """
    try:
        # 获取所有历史记录文件
        history_files = []
        for filename in os.listdir(LOGS_FOLDER):
            if filename.startswith('report_log_') and filename.endswith('.json'):
                history_files.append(filename)
        
        # 按日期排序（最新的在前）
        history_files.sort(reverse=True)
        print(f"找到历史记录文件: {history_files}")
        
        # 读取所有历史记录
        all_records = []
        for log_file in history_files:
            log_path = os.path.join(LOGS_FOLDER, log_file)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    print(f"从 {log_file} 读取到 {len(logs)} 条记录")
                    for log in logs:
                        # 添加文件路径信息便于访问
                        if 'image_filename' in log:
                            log['image_path'] = f"detections/{log['image_filename']}"
                        else:
                            print(f"警告: 记录缺少 image_filename 字段: {log.get('report_id', '未知ID')}")
                        
                        # 如果有PDF文件名，构建PDF路径
                        if 'pdf_filename' in log:
                            pdf_rel_path = os.path.join('reports_pdf', log['pdf_filename'])
                            if os.path.exists(os.path.join(os.path.dirname(__file__), pdf_rel_path)):
                                log['pdf_path'] = pdf_rel_path
                        
                        # 格式化时间为易读格式
                        if 'timestamp' in log:
                            try:
                                # 解析ISO格式时间戳
                                dt = datetime.fromisoformat(log['timestamp'])
                                log['formatted_time'] = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except Exception as e:
                                print(f"时间格式化错误: {e}")
                                log['formatted_time'] = log['timestamp']
                        else:
                            print(f"警告: 记录缺少 timestamp 字段: {log.get('report_id', '未知ID')}")
                        
                        # 检查医生署名
                        if 'doctor_name' not in log or not log['doctor_name']:
                            print(f"警告: 记录缺少医生署名: {log.get('report_id', '未知ID')}")
                        
                        # 检查检测结果
                        if 'detections' not in log:
                            print(f"警告: 记录缺少检测结果: {log.get('report_id', '未知ID')}")
                        
                        all_records.append(log)
            except Exception as e:
                print(f"读取日志文件 {log_file} 失败: {str(e)}")
        
        # 按时间排序，最新的在前
        all_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        print(f"总共读取到 {len(all_records)} 条历史记录")
        
        # 过滤掉没有PDF的重复记录
        filtered_records = []
        image_filenames_with_pdf = set()  # 记录已有PDF的图像文件名
        
        # 首先添加所有有PDF的记录
        for record in all_records:
            if 'pdf_path' in record and record.get('image_filename'):
                filtered_records.append(record)
                image_filenames_with_pdf.add(record.get('image_filename'))
        
        # 然后添加没有与之重复的无PDF记录
        for record in all_records:
            if 'pdf_path' not in record and record.get('image_filename') and record.get('image_filename') not in image_filenames_with_pdf:
                filtered_records.append(record)
        
        # 重新按时间排序
        filtered_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        print(f"过滤后保留 {len(filtered_records)} 条历史记录")
        
        return render_template('history.html', records=filtered_records)
    
    except Exception as e:
        print(f"查看历史记录失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('history.html', records=[], error=f"加载历史记录失败: {str(e)}")


@app.route('/history/detail/<report_id>')
def history_detail(report_id):
    """
    查看单个历史记录的详细信息
    """
    try:
        print(f"正在查找历史记录: {report_id}")
        found_record = None
        
        # 查找对应的记录
        for filename in os.listdir(LOGS_FOLDER):
            if filename.startswith('report_log_') and filename.endswith('.json'):
                log_path = os.path.join(LOGS_FOLDER, filename)
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                        for log in logs:
                            if log.get('report_id') == report_id:
                                print(f"在文件 {filename} 中找到记录 {report_id}")
                                found_record = log
                                
                                # 添加文件路径信息便于访问
                                if 'image_filename' in log:
                                    log['image_path'] = f"detections/{log['image_filename']}"
                                
                                # 如果有PDF文件名，构建PDF路径
                                if 'pdf_filename' in log:
                                    pdf_rel_path = os.path.join('reports_pdf', log['pdf_filename'])
                                    if os.path.exists(os.path.join(os.path.dirname(__file__), pdf_rel_path)):
                                        log['pdf_path'] = pdf_rel_path
                                        print(f"PDF文件存在: {pdf_rel_path}")
                                    else:
                                        print(f"PDF文件不存在: {pdf_rel_path}")
                                else:
                                    print(f"记录没有PDF文件名字段")
                                
                                # 格式化时间为易读格式
                                if 'timestamp' in log:
                                    try:
                                        # 解析ISO格式时间戳
                                        dt = datetime.fromisoformat(log['timestamp'])
                                        log['formatted_time'] = dt.strftime("%Y-%m-%d %H:%M:%S")
                                    except:
                                        log['formatted_time'] = log['timestamp']
                                
                                # 检查记录中的关键字段
                                if 'detections' not in log or not log['detections']:
                                    print(f"警告: 记录缺少检测结果")
                                if 'doctor_name' not in log or not log['doctor_name']:
                                    print(f"警告: 记录缺少医生署名")
                                if 'ai_analysis' not in log:
                                    print(f"警告: 记录缺少AI分析结果")
                                
                                return render_template('history_detail.html', record=log)
                except Exception as e:
                    print(f"读取日志文件 {filename} 出错: {str(e)}")
                    continue
        
        if not found_record:
            print(f"未找到记录 {report_id}")
        
        return render_template('history_detail.html', record=None, error="未找到指定的历史记录")
    
    except Exception as e:
        print(f"查看历史记录详情失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('history_detail.html', record=None, error=f"加载历史记录失败: {str(e)}")


@app.route('/download_history_pdf/<filename>')
def download_history_pdf(filename):
    """
    下载历史记录中的PDF文件
    """
    try:
        pdf_path = os.path.join(REPORTS_PDF_FOLDER, filename)
        if os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': '文件不存在'}), 404
            
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=FLASK_CONFIG['debug'])