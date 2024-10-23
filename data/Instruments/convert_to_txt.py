import json


# jsonfile = json.loads('Instruments.index.json')
# 读取 JSON 文件
with open('Instruments.index.json', 'r') as file:
    jsonfile = json.load(file)

d = {}
with open('index', 'w') as f:
    for key in jsonfile:
        f.write(key + ' ' + ''.join(jsonfile[key]) + '\n')
    f.close()
