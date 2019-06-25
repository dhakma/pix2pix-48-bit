import cv2
import numpy as np
from models.TrajLoss import TrajLoss

class TrajViz:
    def __init__(self):
        self.colors =[(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
        self.tl = TrajLoss()

    def __get_color(self, i):
        return self.colors[i % len(self.colors)]

    def viz_traj(self, trajs, dims=[512, 512]):
        img = 128 * np.ones(np.hstack((dims, 3)), np.uint8)
        # assume all traj points[0,1]
        for i, traj in trajs.items():
            m = np.min(traj, 0)
            ma = np.max(traj, 0)
            for pt in traj:
                # scaled_pt = (pt * np.array(dims) + np.random.rand(2)).astype(np.int)
                scaled_pt = (pt * np.array(dims)).astype(np.int)
                cv2.circle(img, (scaled_pt[0], scaled_pt[1]), 2, self.__get_color(i));
        return img

    def viz_traj_from_vecseq_img(self, vec_img):
        vec_seq = self.tl.im2traj(vec_img, True)
        traj = self.tl.vecseq2traj(vec_seq)
        viz_img = self.viz_traj(traj, [256, 256])
        return viz_img


