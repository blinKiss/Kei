# tqdm 라이브러리 : 반복문 등 사용시 진행사항 표시 기능
from tqdm import tqdm
import time

for x in tqdm(range(100)):
    print(x)
    time.sleep(0.5)