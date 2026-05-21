import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from skimage.util import img_as_ubyte



BASE_PATH = r"C:\Users\dlgkr\OneDrive\Desktop\code\astronomy\asteroid_AI\data_analysis\final_agent"
idx = 325

FOLDER_PATH = "/"+str(idx - idx%100)+"/"

if not os.path.exists(BASE_PATH+FOLDER_PATH):
    raise NameError("No such folder found")

img_list0 = os.listdir(BASE_PATH+FOLDER_PATH)
img_list0 = [x for x in img_list0 if ".png" in x]
img_LC_list = [[]]

p1 = re.compile(r'No\.\d+')
pNum = re.compile(r'\d+')
p2 = re.compile(r'\d+-th LC')
p3 = re.compile(r't = \d+')
for name in img_list0:
    numstr = p1.findall(name)[0]
    num = int(pNum.findall(numstr)[0])
    LCidxstr = p2.findall(name)[0]
    LCidx = int(pNum.findall(LCidxstr)[0])
    if num == (idx%100):
        if LCidx >= len(img_LC_list):
            img_LC_list.append([])
        img_LC_list[LCidx].append(name)

for img_list in img_LC_list:
    img_list.sort(key=lambda x: int(pNum.findall(p3.findall(x)[0])[0]))

gif_config = {
    'loop':0, ## 0으로 세팅하면 무한 반복, 3으로 설정하면 3번 반복
    'duration': 1.0 ## 다음 화면으로 넘어가는 시간
}
 
for i in tqdm(range(len(img_LC_list))):
    ## gif로 만들 이미지를 리스트로 만들어 줌
    images = [plt.imread(BASE_PATH+FOLDER_PATH+x) for x in img_LC_list[i]]

    images = [img_as_ubyte(frame[..., :3]) for frame in images]
 
    ## mimwrite 대신 mimsave로도 가능
    imageio.mimwrite(BASE_PATH+FOLDER_PATH+'Env No.%d %d-th LC.gif'%(idx%100, i), ## 저장 경로
                     images, ## 이미지 리스트
                     format='gif', ## 저장 포맷
                     **gif_config ## 부가 요소
                    )