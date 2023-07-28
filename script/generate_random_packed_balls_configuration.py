import numpy as np


def get_ball_list(radius_low=0.1, radius_up=0.2, tries=4096):
    ball_list = []
    volume = 0.0
    random_data = np.random.rand(tries, 4)
    random_data[:, -1] = radius_low + (radius_up - radius_low) * random_data[:, -1]
    for i in range(tries):
        # Check whether the current ball can be contained in the box.
        # flag = (
        #     random_data[i, -1] <= np.min(random_data[i, :-1])
        #     and np.max(random_data[i, :-1]) <= 1.0 - random_data[i, -1]
        # )
        flag = True
        new_ball = [
            random_data[i, 0],
            random_data[i, 1],
            random_data[i, 2],
            random_data[i, 3],
        ]
        if not flag:
            continue
        # Check whether any two balls intersect.
        for ball in ball_list:
            distance = np.linalg.norm(np.array(ball[:-1]) - np.array(new_ball[:-1]))
            if distance < ball[-1] + new_ball[-1]:
                flag = False
                break
        if not flag:
            continue
        ball_list.append(new_ball)
        volume += 4 / 3 * np.pi * new_ball[-1] ** 3
        print(
            "Add a new ball, current balls={0:d}, volume={1:.6e}".format(
                len(ball_list), volume
            )
        )

    return ball_list


ball_list = get_ball_list(radius_low=0.01, radius_up=0.05, tries=4096)
data_to_write = np.array(ball_list)
data_to_write.tofile("bin/ball-list-{:d}-r1.bin".format(len(ball_list)))

ball_list = get_ball_list(radius_low=0.05, radius_up=0.1, tries=4096)
data_to_write = np.array(ball_list)
data_to_write.tofile("bin/ball-list-{:d}-r2.bin".format(len(ball_list)))

ball_list = get_ball_list(radius_low=0.1, radius_up=0.2, tries=4096)
data_to_write = np.array(ball_list)
data_to_write.tofile("bin/ball-list-{:d}-r3.bin".format(len(ball_list)))
