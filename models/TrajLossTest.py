from models.TrajLoss import TrajLoss
import numpy as np
from util import util, html

if __name__ == '__main__':
    tl = TrajLoss()
    trajs = tl.gen_random_traj(ppl=50)
    # print(trajs)
    img = tl.traj2im(trajs)
    util.save_image_cv2(img, 'test.png')
    decoded_trajs = tl.im2traj(img, True)
    for k, traj in trajs.items():
        if k not in decoded_trajs: continue
        if not np.allclose(traj, decoded_trajs[k], 1e-5, 1e-4):
            print(np.hstack((traj, decoded_trajs[k])))

    v1 = tl.traj2vecseq(trajs)
    # normalized_v1 = tl.normalize(v1)
    v2 = tl.normalize(tl.traj2vecseq(decoded_trajs))
    print('loss : ', tl.calc_vecseqloss(v2, v2))
    util.save_image_cv2(tl.traj2im(v1), 'test_vec.png');

    start_poss = tl.extract_start_pos(trajs)
    retrieved_trajs = tl.vecseq2traj(v1, start_poss);
    for k, traj in trajs.items():
        if not np.allclose(traj, retrieved_trajs[k], 1e-5, 1e-4):
            print(np.hstack((traj, retrieved_trajs[k])))
    util.save_image_cv2(tl.traj2im(retrieved_trajs), 'test-vec2traj.png')




