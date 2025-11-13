import cv2
import base64
import time
import os
from openai import OpenAI

# 全局客户端实例
client = None

def initialize_client(api_key):
    """初始化OpenAI客户端"""
    global client
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        print("警告: OpenAI API Key未设置。LLM功能将不可用。")
        client = None
        return
    try:
        client = OpenAI(api_key=api_key)
        # 尝试一次简单的API调用来验证key
        client.models.list()
        print("✅ OpenAI客户端初始化成功。")
    except Exception as e:
        print(f"❌ OpenAI客户端初始化失败: {e}")
        print("   请检查您的API Key是否正确或网络连接是否正常。")
        client = None


def encode_image_to_base64(image_data):
    """将OpenCV图像数据（numpy数组）编码为base64字符串"""
    success, buffer = cv2.imencode('.jpg', image_data)
    if not success:
        print("❌ 图像编码失败。")
        return None
    return base64.b64encode(buffer).decode('utf-8')


def draw_bbox_on_image(image, bbox, case_id=None):
    """
    在图像上绘制一个边界框。
    :param image: OpenCV图像 (numpy array)。
    :param bbox: 边界框坐标 [x_min, y_min, x_max, y_max]。
    :return: 绘制了边界框的图像 (numpy array)。
    """
    img_with_bbox = image.copy()
    x1, y1, x2, y2 = bbox
    # 使用亮绿色和较粗的线条，使其更显眼
    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if case_id is not None:
        save_dir = "./pipeline_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{case_id}_with_bbox.jpg")
        cv2.imwrite(save_path, img_with_bbox)
        # 可选：展示图像
        # cv2.imshow(f"bbox_case_{case_id}", img_with_bbox)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return img_with_bbox


def get_gpt4o_response(image_with_bbox, question, model="gpt-4o"):
    """
    调用GPT-4o API获取对带BBox图像的回答。
    :param image_with_bbox: 带有BBox的OpenCV图像 (numpy array)。
    :param question: 用户的文本问题。
    :param model: 使用的GPT模型。
    :return: GPT-4o的回答字符串，或出错时的错误信息。
    """
    if client is None:
        return "Error: OpenAI client not initialized."

    base64_image = encode_image_to_base64(image_with_bbox)
    if base64_image is None:
        return "Error: Failed to encode image."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    rounds = 0
    while rounds < 3:
        rounds += 1
        try:
            print("📡 正在调用GPT-4o API...")
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                n=1,
                max_tokens=300, # 限制最大输出长度
            )
            content = response.choices[0].message.content
            end_time = time.time()
            print(f"✅ GPT-4o回答成功！(耗时: {end_time - start_time:.2f} 秒)")
            return content.strip()
        except Exception as e:
            print(f"⚠️ GPT-4o API调用错误 (第{rounds}次尝试): {e}")
            time.sleep(5)

    return f"Error: GPT-4o API failed after {rounds} retries."