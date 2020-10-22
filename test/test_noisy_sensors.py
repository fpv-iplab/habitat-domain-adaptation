import os
import random

import numpy as np
import pytest
import quaternion
import matplotlib.pyplot as plt

import habitat
from habitat.config.default import get_config
from habitat.tasks.nav.nav import (
    MoveForwardAction,
    TurnLeftAction,
    NavigationEpisode,
    NavigationGoal,
)
from habitat.tasks.utils import quaternion_rotate_vector
from habitat.utils.geometry_utils import angle_between_quaternions
from habitat.utils.test_utils import sample_non_stop_action
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
)

def test_noise():
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = ("data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH=5.0
    config.SIMULATOR.TURN_ANGLE = 10
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25

    
    config.SIMULATOR.ACTION_SPACE_CONFIG = "customrobotnoisy"
    config.SIMULATOR.NOISE_MODEL = habitat.Config()
    config.SIMULATOR.NOISE_MODEL.ROBOT = "Universal"
    config.SIMULATOR.NOISE_MODEL.CONTROLLER = "Proportional"
    config.SIMULATOR.NOISE_MODEL.NOISE_STD = 0.05
    config.SIMULATOR.NOISE_MODEL.ROT_NOISE_STD = 5.0
    
    config.TASK.SENSORS = ["COMPASS_SENSOR", "GPS_SENSOR", "POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    config.TASK.GPS_SENSOR.DIMENSIONALITY = 2
    #config.TASK.GPS_SENSOR.GPS_NOISE_AMOUNT = 0.1
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GPS_NOISE_AMOUNT = 0.1
    #config.TASK.COMPASS_SENSOR.ROT_NOISE_AMOUNT = 10.0
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.ROT_NOISE_AMOUNT = 10.0

    config.freeze()

    valid_start_position = [3.0, 0.10, 20.0] # y is the height

    expected_pointgoal = [6.0, 0.10, 5.0]
    goal_position = np.add(valid_start_position, expected_pointgoal)

    # starting quaternion is rotated by 180 degrees along z-axis, which
    # corresponds to simulator using z-negative as forward action
    start_rotation = [0, 0, 0, 1]
    test_episode = NavigationEpisode(
        episode_id="0",
        scene_id=config.SIMULATOR.SCENE,
        start_position=valid_start_position,
        start_rotation=start_rotation,
        goals=[NavigationGoal(position=goal_position)],
    )
    env = habitat.Env(config=config, dataset=None)

    env.episode_iterator = iter([test_episode])
    env.reset()

    N_STEPS = 20
    actions = [sample_non_stop_action(env.action_space) for _ in range(N_STEPS)]

    #-----------------------------------------------------------------
    old_pos=[0,0]
    old_rot=0

    display=False

    for action in actions:
        obs=env.step(action=MoveForwardAction.name)
        #obs=env.step(action=TurnLeftAction.name)
        
        
        rgb = obs["rgb"]
        depth = obs["depth"]
        gps = obs["gps"]
        rotation = obs["compass"]
        
        if display:
            plt.figure(figsize=(10,10))
            plt.subplot(1,2,1)
            plt.title(str(gps)+","+str(np.rad2deg(rotation))+
                    ", relative mov.:"+str(gps-old_pos)+","+str(np.rad2deg(rotation-old_rot)))
            plt.imshow(rgb)
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(np.squeeze(depth), cmap='gray_r')
            plt.axis('off')
            plt.show()
        else:
            print( "GPS:",gps,", rotation:",np.rad2deg(rotation),", relative mov.:",gps-old_pos,", ",np.rad2deg(rotation-old_rot) )
        old_pos=gps
        old_rot=rotation
        
    env.close()

if __name__=="__main__":
    test_noise()