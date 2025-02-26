import requests
import math
import time
import random
import json
import tqdm
fake_id = 'MzU4MjAzNzU2OQ=='
token = 656908361

url = "https://mp.weixin.qq.com/cgi-bin/appmsg"
cookie = 'ua_id=I5u3i0sgmB0XNk2KAAAAAIZtl17FO39nrXVmuB_wZcA=; wxuin=27584358332597; _clck=ui7dzj|1|fpl|0; uuid=c77bb4bacd73b4f29cada6428df4a435; rand_info=CAESIF+gouX2RG3RlIGfH2MsAXAOWFCqjVHJWpnqyZ8pnJKN; slave_bizuin=3893182116; data_bizuin=3893182116; bizuin=3893182116; data_ticket=rmdxHFEsyBuP/WfyV9FtufJx4TRQdZLjSj3Ppy3J/8rfZ2XNUiZBWZP+5BH2kIHY; slave_sid=OWVzdFFNd1ZWc3pTNFFuWWNJbDVNdHlSTVYyRXY5VVQ3WWZBdTVKUHNJUDR6ZXI2eGFEdFJwamVuOEJNdzJTZFV1VXpuQXdWOFFydnVpdGNMTVFoRkZnaUw5aFE2NW1PRk5XY0k3MWVBRUhGY0FkekZHQWF3YjFiZGlVTUlXNUtNc3VlUGRRQzNqNTBTdjh1; slave_user=gh_01a569175ff9; xid=1a24d002d259e5dc019a35ad2a5b5cdc; mm_lang=zh_CN; _clsk=nbkhqo|1727584450146|3|1|mp.weixin.qq.com/weheat-agent/payload/record'


headers = {
    "Cookie": cookie,
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Mobile Safari/537.36",
}

params = {
    "token": token,
    "lang": "zh_CN",
    "f": "json",
    "ajax": "1",
    "action": "list_ex",
    "begin": "0",
    "count": "5",
    "query": "",
    "fakeid": fake_id, # 自己的号，设置为空
    "type": "9",
}

# app_msg_list = []
# # 在不知道公众号有多少文章的情况下，使用while语句
# # 也方便重新运行时设置页数
# with open("app_msg_list.csv", "w", encoding='utf-8') as file:
#     file.write("文章标识符aid,标题title,链接url,时间time\n")

# i = 0
# while True:
#     begin = i * 5
#     params["begin"] = str(begin)
#     # 随机暂停几秒，避免过快的请求导致被查到
#     # time.sleep(random.randint(1, 10))
#     resp = requests.get(url, headers=headers, params=params, verify=False)
#     # 微信流量控制, 退出
#     if resp.json()['base_resp']['ret'] == 200013:
#         print("Frequency control, stop at {}".format(str(begin)))
#         time.sleep(3600)
#         continue
    
#     # 如果返回的内容中为空则结束
#     if len(resp.json()['app_msg_list']) == 0:
#         print("All articles parsed")
#         break
        
#     msg = resp.json()
#     if "app_msg_list" in msg:
#         for item in msg["app_msg_list"]:
#             info = '"{}","{}","{}","{}"'.format(str(item["aid"]), item['title'], item['link'], str(item['create_time']))
#             with open("app_msg_list.csv", "a", encoding='utf-8') as f:
#                 f.write(info + '\n')
#         print(f"Page {i} successfully crawled\n")
#         print("\n".join(info.split(",")))
#         print("\n\n---------------------------------------------------------------------------------\n")

#     # 翻页
#     i += 1    

with open("app_msg_list.csv","r",encoding="utf-8") as f:
    data = f.readlines()
n = len(data)
print(n)
for i in range(n):
    mes = data[i].strip("\n").split(",")
    if len(mes)!=4:
        print(mes)
        continue
    title,url = mes[1:3]
    if i>0:
        r = requests.get(eval(url),headers=headers)
        if r.status_code == 200:
            text = r.text
            print(text)
        break
