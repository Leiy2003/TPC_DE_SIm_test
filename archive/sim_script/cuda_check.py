import time
import os

# 检查的间隔
interval = 2
# 连续检查的时间，单位是秒
continuous_time = 60
# 连续检查的次数
check_times = continuous_time // interval
# 文件路径
file_path = "cuda_status.txt"

while True:
    with open(file_path, "r+") as f:
        contents = f.read()

    if contents.strip() == "1":
        check_times -= 1
    else:
        check_times = continuous_time // interval

    if check_times == 0:
        with open(file_path, "w") as f:
            f.write("0")
        check_times = continuous_time // interval

    time.sleep(interval)
