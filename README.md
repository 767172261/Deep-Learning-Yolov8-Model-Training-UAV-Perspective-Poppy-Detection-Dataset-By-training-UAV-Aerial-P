深度学习Yolov8模型 训练无人机视角罂粟检测数据集 通过训练出的无人机航拍罂粟检测数据集  建立基于深度学习Yolov8罂粟识别检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f397055317e9426289229773e8b26677.png)


以下文字及代码等仅供参考学习。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b34ffe0ea18f453aaf76e37d84c68bfb.png)


@[toc]
无人机航拍视角下粟识别检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a5c264243c64f79855aa22f67a3e009.png)

数据类型：Image+YOLO标签![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a8d99f9ede934f5e919156adb929b002.jpeg)

（3160张）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4734fa5c44c45a889882c441b473f1c.jpeg)

类别：Poppy
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ecb73b196eeb4843be1aa27732ce3a16.png)
使用YOLOv8训练基于无人机航拍视角下粟（Poppy）识别检测的数据集，步骤进行：数据集准备与预处理、划分数据集、创建`data.yaml`文件、环境搭建、模型训练、推理与部署、性能评估以及模型导出。以下是每个步骤的详细指南和代码示例。
代码示例，仅供参考学习。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f95dd409235f4c88a6029416e997900f.png)


### 1. 数据集准备与预处理
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2c4c4fb80e0b449aa7459d6f17221c72.png)

假设你的数据集已经按照以下结构组织：

```
poppyDataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

确保所有的图像和标签文件都已正确命名，并且每张图像都有对应的YOLO格式的标注文件（`.txt`）。

### 2. 划分数据集

数据集还没有划分成训练集、验证集和测试集，可以使用Python脚本进行划分。这里我们按照7:2:1的比例进行划分。

```python
import os
import random
import shutil

def split_dataset(source_dir, dest_train, dest_val, dest_test, ratio=(0.7, 0.2, 0.1)):
    images = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.jpg')]
    random.shuffle(images)
    
    total = len(images)
    train_size = int(total * ratio[0])
    val_size = int(total * ratio[1])
    
    for i, image_file in enumerate(images):
        label_file = image_file.replace('.jpg', '.txt')
        
        if i < train_size:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_train, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_train, 'labels', label_file))
        elif i < train_size + val_size:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_val, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_val, 'labels', label_file))
        else:
            shutil.move(os.path.join(source_dir, 'images', image_file), os.path.join(dest_test, 'images', image_file))
            shutil.move(os.path.join(source_dir, 'labels', label_file), os.path.join(dest_test, 'labels', label_file))

# 示例调用
split_dataset('path/to/source/poppyDataset', 
              'path/to/dest/poppyDataset/train',
              'path/to/dest/poppyDataset/val',
              'path/to/dest/poppyDataset/test')
```

### 3. 创建 `data.yaml` 文件

在项目根目录下创建 `data.yaml` 文件，内容如下：

```yaml
train: ./poppyDataset/images/train
val: ./poppyDataset/images/val
test: ./poppyDataset/images/test

nc: 1 # 类别数量
names: ['Poppy'] # 类别名称
```

### 4. 环境搭建

确保安装了Ultralytics YOLOv8和其他必要的依赖库：

```bash
pip install ultralytics
```

### 5. 模型训练

开始训练过程：

```bash
yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=100 imgsz=640 batch=16
```

### 6. 推理与部署

完成训练后，你可以加载最佳权重文件来进行推理。下面是一个简单的推理脚本示例：

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/exp/weights/best.pt')

def detect_poppy(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = 'Poppy'
            print(f"Detected {label} at ({x1}, {y1}) - ({x2}, {y2}), Confidence: {confidence}")
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    cv2.imshow('Poppy Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 7. 性能评估

使用验证集或测试集评估模型性能：

```bash
yolo task=detect mode=val model=runs/detect/exp/weights/best.pt data=path/to/data.yaml
```

### 8. 模型导出

训练完成后，如果需要将模型导出为其他格式（如ONNX），可以使用以下命令：

```bash
yolo export model=runs/detect/exp/weights/best.pt format=onnx
```


然后根据训练的权重推理识别，建立深度学习罂粟识别检测系统

从，包括环境搭建、模型加载、推理过程实现以及系统的部署。这里以YOLOv8为例，介绍如何基于训练好的模型来构建一个罂粟识别检测系统。假设你已经有了针对无人机变焦航拍视角下罂粟（Poppy）识别训练好的模型。

### 1. 环境搭建

确保你的环境中安装了必要的依赖库。可以使用以下命令安装Ultralytics YOLOv8和其他需要的库：

```bash
pip install ultralytics opencv-python-headless
```

`opencv-python-headless` 是一个轻量级版本的OpenCV，适用于不需要GUI操作的场景。

### 2. 加载预训练模型

首先加载你已经训练好的模型（例如`best.pt`），用于进行罂粟识别检测。

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('path/to/your/best.pt')
```

### 3. 实现推理过程

编写代码以处理输入图像，并使用模型进行预测。下面是一个简单的例子，演示如何处理单张图像：

```python
import cv2

def detect_poppy(image_path):
    # 加载图像
    img = cv2.imread(image_path)
    
    # 使用模型进行预测
    results = model(img)  # 返回的结果包含边界框、置信度等信息
    
    # 遍历每个检测结果
    for result in results:
        boxes = result.boxes  # 获取所有检测到的对象的边界框
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = 'Poppy'
            print(f"Detected {label} at ({x1}, {y1}) - ({x2}, {y2}), Confidence: {confidence}")
            
            # 在原图上绘制边界框和标签（可选）
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示结果图像（可选）
    cv2.imshow('Poppy Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

需要实时处理视频流，可以对上述代码稍作修改，以从摄像头或视频文件中读取帧并进行检测：

```python
cap = cv2.VideoCapture('video.mp4')  # 或者使用数字索引如0来捕获摄像头视频流

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用模型进行预测
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
            label = 'Poppy'
            
            # 绘制边界框和标签
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 显示帧
    cv2.imshow('Poppy Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. 系统部署

为了将这个检测系统部署到实际应用中，你可以考虑多种部署方式：

- **Web服务**：使用Flask或Django等框架创建一个Web服务，通过API接收图片或视频流并返回检测结果。
  
  示例：使用Flask创建一个简单的Web服务
  
  ```python
  from flask import Flask, request, jsonify
  import numpy as np
  from PIL import Image
  import io
  
  app = Flask(__name__)
  
  @app.route('/detect', methods=['POST'])
  def detect():
      if 'image' not in request.files:
          return jsonify({'error': 'No image provided'}), 400
      
      file = request.files['image']
      img_bytes = file.read()
      img = Image.open(io.BytesIO(img_bytes))
      
      # 转换为适合YOLO模型输入的格式
      img_array = np.array(img)
      
      # 使用模型进行预测
      results = model(img_array)
      
      detections = []
      for result in results:
          for box in result.boxes:
              x1, y1, x2, y2, confidence, class_id = box.xyxy.tolist() + [box.conf.item(), int(box.cls.item())]
              detections.append({
                  'bbox': [x1, y1, x2, y2],
                  'confidence': confidence,
                  'class': 'Poppy'
              })
      
      return jsonify(detections)
  
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)
  ```



### 总结

、这样打架作为一个参考和学习，如何基于YOLOv8训练好的模型建立一个罂粟识别检测系统，、仅供参考学习使用。

