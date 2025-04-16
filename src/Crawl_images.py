import requests
from bs4 import BeautifulSoup
import os

# 目标网页 URL
url = 'https://pixnio.com/zh/'

# 请求头，模拟浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 存储图片的目录
save_path = r'D:\桌面\photo'

# 确保存储图片的目录存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

try:
    # 发送请求获取网页内容
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # 检查请求是否成功

    # 解析 HTML 页面
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有图片标签
    img_tags = soup.find_all('img')

    # 遍历图片标签
    for index, img_tag in enumerate(img_tags, start=1):
        # 获取图片的 src 属性
        img_url = img_tag.get('src')
        if img_url:
            # 处理相对 URL
            if not img_url.startswith('http'):
                img_url = url.rstrip('/') + '/' + img_url.lstrip('/')

            try:
                # 发送请求获取图片内容
                img_response = requests.get(img_url, headers=headers)
                img_response.raise_for_status()  # 检查请求是否成功

                # 生成图片文件名
                img_name = os.path.join(save_path, f'{index}.jpg')

                # 保存图片到本地
                with open(img_name, 'wb') as f:
                    f.write(img_response.content)
                print(f'成功保存图片: {img_name}')
            except requests.RequestException as e:
                print(f'请求图片 {img_url} 失败: {e}')
except requests.RequestException as e:
    print(f'请求页面失败: {e}')
except Exception as e:
    print(f'发生未知错误: {e}')