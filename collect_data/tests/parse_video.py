import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import progressbar
from npy_append_array import NpyAppendArray


def main(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    #video_name = 'YouTube Rewind 2019 For the Record  YouTubeRewind.mp4'
    #video_name = 'Hearthstone Classic Zoo on EU!-download.mp4'
    video_name = 'Hearthstone Classic Zoo on EU!.mp4'
    video_path = os.path.join(dir_path, '..', '..', 'raw_videos', video_name)
    data_folder = os.path.join(dir_path, '..', '..', 'data', args.data_folder)
    data_file_prefix = os.path.join(data_folder, args.output)

    print(video_path)
    assert Path(video_path).exists, f'path {video_path} does not exist'

    # import imageio

    # frames = []
    # vid = imageio.get_reader(video_path,  'ffmpeg')
    # import pdb; pdb.set_trace()
    # image = vid.get_data(0)


    
    vidcap = cv2.VideoCapture(video_path)
    video_length_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    real_fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    video_length_seconds = video_length_frames / real_fps

    def crop_resize_img(img):
        factor = 3 # 3 for 360p
        left = 280 / factor
        width = 1400 / factor
        height = 1080 / factor
        img = img.crop((left, 0, left + width, height))

        scale = 10.9375 / factor
        img = img.resize((round(img.size[0]/scale), round(img.size[1]/scale)))

        return img

    def get_frame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if not hasFrames:
            return hasFrames, image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = crop_resize_img(image)
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)
        # if hasFrames:
        #     cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames, image

    sec = 0
    frame_rate = 4
    real_fps / frame_rate
    second_increment = 1 / frame_rate # capture every .25 second
    count=1
    success, image = get_frame(sec)

    #images = []

    
    widgets = [' [',
            progressbar.Timer(),
            '] ',
            progressbar.Bar('*'),' (',
            progressbar.ETA(), ') ',
            ]
    
    bar = progressbar.ProgressBar(maxval=video_length_seconds, 
                                widgets=widgets).start()

    while success:
        sec = sec + second_increment
        sec = round(sec, 2)
        success,image = get_frame(sec)
        if not success:
            print(sec)
            break
        with NpyAppendArray(data_file_prefix + '.npy') as npaa:
            npaa.append(image)
        #bar.update(count*frame_rate)
        bar.update(sec)
        count = count + 1

    
    # states = np.stack(images)
    # np.save(open(data_file_prefix + '.npy', 'wb'), states)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True, type=str)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    main(args)
