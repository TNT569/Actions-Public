import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

#加载图片与视频

source_image = imageio.imread('test.jpg')#最好使用jpg格式，png慢很多也并不会有更好的效果
driving_video = imageio.get_reader('damedane.mp4')
#可以使用其他人脸视频替换damedane


#将图片与视频尺寸转换为 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

from skimage import img_as_ubyte
from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='vox-cpk.pth.tar',cpu = False)#如能使用gpu，则将参数换位cpu=false
from demo import make_animation
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False, adapt_movement_scale=True,cpu = True)
imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions])#修改generated.mp4为你想输出的视频名称
print('success')
