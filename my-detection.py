import jetson.inference
import jetson.utils
import cv2
import os

# 确保保存图像的文件夹存在
save_dir = 'detections'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 加载预训练模型
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 加载图像
img = jetson.utils.loadImage('test_pic.jpeg')

# 使用模型检测图像中的对象
detections = net.Detect(img)

# 将cudaImage转换为OpenCV图像格式
image = jetson.utils.cudaToNumpy(img)
height, width, _ = image.shape

# 遍历检测结果，并输出详细信息
for i, detection in enumerate(detections):
    # 获取对象的类别ID和置信度
    class_id = detection.ClassID
    confidence = detection.Confidence
    
    # 获取对象的边界框坐标
    left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
    
    # 计算宽度、高度和面积
    bbox_width = right - left
    bbox_height = bottom - top
    bbox_area = bbox_width * bbox_height
    
    # 计算中心点坐标
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    # 获取对象的类别名称
    class_name = net.GetClassDesc(class_id)
    
    # 打印对象的详细信息
    print(f"Detected: {class_name}, Confidence: {confidence:.2f}")
    print(f"Coordinates: Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")
    print(f"Width: {bbox_width}, Height: {bbox_height}, Area: {bbox_area}")
    print(f"Center: ({center_x}, {center_y})")
    
    # 裁剪检测到的对象并保存为JPEG格式
    roi = image[top:bottom, left:right]
    cv2.imwrite(os.path.join(save_dir, f'{class_name}_{i}.jpg'), roi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# 在原始图像上绘制边界框和类别名称
for detection in detections:
    left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
    
    class_name = net.GetClassDesc(detection.ClassID)
    
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} {detection.Confidence:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 保存标注好的完整图像为JPEG格式
cv2.imwrite('annotated_image.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# 如果需要显示图像和检测结果，可以使用以下代码
# display = jetson.utils.videoOutput("display://0")
# display.Render(img)
# display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))