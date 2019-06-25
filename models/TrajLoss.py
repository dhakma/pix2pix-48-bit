import math

import numpy as np
from util import util, html

class TrajLoss:

    def __init__(self):
        # self.output = output
        self.pxl_size = np.array([1, 8])
        self.img_size = np.array([256, 256])
        self.border_width = self.img_size[0] - 224

    # def split_img_to_x_and_y(self, img):
    #     width, height, channels = img.shape
    #     img1 = mask[1:width]

    def im2traj(self, img, is_16_bit):
        pxl_size = self.pxl_size
        pxl_area = pxl_size[0] * pxl_size[1]

        row_step_size = pxl_size[1]

        img_size = self.img_size
        border_width = self.border_width

        traj = {}

        for y in range(border_width, img_size[1] - border_width, row_step_size):
            x = border_width
            end_x = img_size[0] - border_width

            agent_id = int((y - border_width) / row_step_size)
            traj[agent_id] = list()

            while (x < end_x):
                agg_pos_val = np.array([0, 0, 0]).astype(np.float32)

                for pxl_x in range(0, pxl_size[0]):
                    for pxl_y in range(0, pxl_size[1]):
                        # agg_pos_val += img[x + pxl_x, y + pxl_y]
                        agg_pos_val += img[y + pxl_y, x + pxl_x]

                x += pxl_size[0]

                agg_pos_val /= (float(pxl_area) * (2 ** 16 - 1))

                traj[agent_id].append(np.asarray([agg_pos_val[0], agg_pos_val[1]]))

        for k, v in traj.items():
            traj[k] = np.asarray(v)

        return traj

    def traj2vecseq(self, traj):
        vecseq = {}
        for id, t in traj.items():
            if len(t) > 1:
                # vecseq[id] = t[1:] - t[:-1]
                s = t[0]
                vecseq[id] = np.vstack((t[0], t[1:] - t[:-1]))
        return self.normalize(vecseq)

    def vecseq2traj(self, vecseq):
        trajs = {}
        denorm_vecseq = self.denormalize(vecseq)
        for id, seq in denorm_vecseq.items():
            traj = []
            if (len(seq) > 0):
                start_pos = seq[0]
                traj = [start_pos]
                prev_pos = start_pos
                for i in range(1, len(seq)):
                    nxt_pos = prev_pos + seq[i]
                    traj.append(nxt_pos)
                    prev_pos = nxt_pos
            trajs[id] = np.array(traj)
        return trajs

    # def vecseq2traj(self, vecseq, start_positions):
    #     trajs = {}
    #     denorm_vecseq = self.denormalize(vecseq)
    #     for id, seq in denorm_vecseq.items():
    #         if id not in start_positions:
    #             raise ValueError(id + 'is missing. id has to be in start positions')
    #         start_pos = start_positions[id]
    #         traj = [start_pos]
    #         prev_pos = start_pos
    #         for i in range(len(seq)):
    #             nxt_pos = prev_pos + seq[i]
    #             traj.append(nxt_pos)
    #             prev_pos = nxt_pos
    #         trajs[id] = np.array(traj)
    #     return trajs

    def calc_vecseqloss(self, vseq1, vseq2):
        sum = 0
        for i, v in vseq1.items():
            v1 = v
            v2 = vseq2[i] if i in vseq2 else np.zeros_like(v1)
            sum += self.calc_vectorloss(v1, v2)
        return sum

    def calc_vectorloss(self, v1, v2):
        diff = v1 - v2
        return np.sum(np.sqrt(np.sum(diff ** 2, 1)))
        # return np.sum(v1 * v2)

    def traj2im(self, traj):

        pxl_size = self.pxl_size
        pxl_area = pxl_size[0] * pxl_size[1]

        row_step_size = pxl_size[1]

        img_size = self.img_size
        border_width = self.border_width

        img = np.zeros((img_size[0], img_size[1], 3), np.uint16)

        for y in range(border_width, img_size[1] - border_width, row_step_size):
            x = border_width
            end_x = img_size[0] - border_width

            pos_written = 0
            agent_id = (y - border_width) / row_step_size

            if agent_id not in traj:
                continue

            while (x < end_x):
                num_steps = len(traj[agent_id])
                if (num_steps <= pos_written):
                    break

                agg_pos_val = np.array([0, 0, 0])
                curr_pos = traj[agent_id][pos_written] * (2 ** 16 - 1)
                # print(curr_pos)

                for pxl_x in range(0, pxl_size[0]):
                    for pxl_y in range(0, pxl_size[1]):
                        # img[x + pxl_x, y + pxl_y] = np.concatenate(([0], curr_pos))
                        # img[x + pxl_x, y + pxl_y] = np.concatenate(([0.0, 0.0], [curr_pos[1]]))
                        img[y + pxl_y, x + pxl_x] = np.concatenate((curr_pos, [0.0]))

                x += pxl_size[0]
                pos_written += 1

        return img

    def normalize(self, trajs):
        traj_min = np.full((2), np.inf)
        traj_max = np.full((2), -np.inf)
        for id, traj in trajs.items():
            traj_min = np.min((traj_min, np.min(traj, axis=0)), axis=0)
            traj_max = np.max((traj_max, np.max(traj, axis=0)), axis=0)

        t1 = trajs[0] - traj_min


        normalized_trajs = {}

        for id, traj in trajs.items():
            # new_traj = []
            # for pt in traj:
            #     new_pt = np.array([.5, .5]) + np.array([.5, .5]) * pt
            #     new_traj.append(new_pt)

            tt = np.array([.5, .5]) + np.array([.5, .5]) * traj
            normalized_trajs[id] = tt
            # np.allclose(trajs[id], tt)

        return normalized_trajs

    def denormalize(self, trajs):
        denoramlized_trajs = {}
        for id, traj in trajs.items():
            denoramlized_trajs[id] = np.array([2., 2.]) * traj - np.array([1., 1.])
        return denoramlized_trajs



    def gen_random_traj(self, ppl=50, steps=192):
        trajs = {}
        # x = np.linspace(-np.pi, np.pi, steps)
        x = np.linspace(-4.0, 4.0, steps)
        x_actual = np.linspace(0, 1, steps)
        y = .5 + .5 * np.sin(x)
        for agent in range(ppl):
            offset = np.full(x.shape, np.random.rand(1))
            # y_offset = y + offset
            y_offset = y
            trajs[agent] = np.array(np.vstack((x_actual, y_offset)).T)
        return trajs

    def extract_start_pos(self, trajs):
        start_pos = {}
        for id, traj in trajs.items():
            start_pos[id] = np.array(traj[0])

        return start_pos

