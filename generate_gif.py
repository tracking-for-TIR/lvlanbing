import imageio
import os
def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def main(path, start, num, name):
    # filelist中的顺序为生成gif的顺序，注意
    filelist = os.listdir(path)
    imagelist = []
    for i in range(start, num):
        imagelist.append(os.path.join(path, filelist[i]))

    gif_name = name
    # 每张图片之间的时间间隔
    duration = 0.05
    create_gif(imagelist, gif_name, duration)

if __name__ == "__main__":
    path = r"G:\my_data\LSOTB-TIR\Evaluation Dataset\sequences\car_S_003\img"
    # 开始图片的位置
    start = 0
    # 生成gif需要图像的数量
    num = 105
    name = "car.gif"
    main(path, start, num, name)