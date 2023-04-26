import os
import re
import imageio
import matplotlib.pyplot as plt

results_dir = 'results/summer2winter_yosemite_0'

frames = []
imgs = sorted(os.listdir(f'{results_dir}/img'), key=lambda x: int(re.sub('\D', '', x)))
for img in imgs:
    img_array = plt.imread(f'{results_dir}/img/{img}') * 255
    frames.append(img_array.astype('uint8'))

with imageio.get_writer(f'{results_dir}/train.gif', mode='I') as writer:
    for frame in frames:
        writer.append_data(frame)