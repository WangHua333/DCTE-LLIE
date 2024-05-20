import os
import shutil
import time

N = 10
for i in range(1, N):
    print(f'当前为第{i}轮实验!')
    if os.path.exists('./checkpoint'):
        print('checkpoint目录存在，正在删除以开始实验...')
        shutil.rmtree('./checkpoint')
        print('删除成功，开始实验！')
    time.sleep(10)
    os.system('python main.py')
