# _*_ coding : utf-8 _*_
# @Time: 2025/7/1 14:43
# @Author : SFnothing
# @File : test
# @Project : BTD
# @Function
import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/QwQ-32B",
    "messages": [
        {
            "role": "user",
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?"
        }
    ]
}
headers = {
    "Authorization": "Bearer sk-frtbnruyvpvlakdkkgzxiqkwhupjkiwxiybpzjoimnlmblpz",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)