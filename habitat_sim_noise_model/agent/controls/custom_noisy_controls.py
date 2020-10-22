#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import attr
import magnum as mn
import numpy as np
import scipy.stats

import habitat_sim.bindings as hsim
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.registry import registry


@attr.s(auto_attribs=True)
class _TruncatedMultivariateGaussian:
    mean: np.array
    cov: np.array

    def __attrs_post_init__(self):
        self.mean = np.array(self.mean)
        self.cov = np.array(self.cov)
        if len(self.cov.shape) == 1:
            self.cov = np.diag(self.cov)

        assert (
            np.count_nonzero(self.cov - np.diag(np.diagonal(self.cov))) == 0
        ), "Only supports diagonal covariance"

    def sample(self, truncation=None):
        if truncation is not None:
            assert len(truncation) == len(self.mean)

        sample = np.zeros_like(self.mean)
        for i in range(len(self.mean)):
            stdev = np.sqrt(self.cov[i, i])
            mean = self.mean[i]
            # Always truncate to 1st standard deviation
            a, b = -1, 1
            sample[i] = scipy.stats.truncnorm.rvs(a, b, mean, stdev)
        return sample


@attr.s(auto_attribs=True)
class MotionNoiseModel:
    linear: _TruncatedMultivariateGaussian
    rotation: _TruncatedMultivariateGaussian


@attr.s(auto_attribs=True)
class ControllerNoiseModel:
    linear_motion: MotionNoiseModel
    rotational_motion: MotionNoiseModel


@attr.s(auto_attribs=True)
class RobotNoiseModel:
    Proportional: ControllerNoiseModel

    def __getitem__(self, key):
        return getattr(self, key)

#pyrobot_noise_models = {
custom_noise_models = {
    "Universal": RobotNoiseModel(
        Proportional=ControllerNoiseModel(
            linear_motion=MotionNoiseModel(
                # parameters for movement noise in case of straight (linear) action
                # [mean_z, mean_x], [var_z, var_x]
                _TruncatedMultivariateGaussian([0.0, 0.0], [1.0, 0.0]), # 0 mean, 1 var
                # parameters for rotational noise in case of straight (linear) action
                _TruncatedMultivariateGaussian([0.0], [1.0]), # 0° +-1° std
            ),
            rotational_motion=MotionNoiseModel(
                # parameters for movement noise in case of rotational action
                _TruncatedMultivariateGaussian([0.0, 0.0], [0.0, 0.0]),
                # parameters for rotational noise in case of rotational action
                _TruncatedMultivariateGaussian([0.0], [1.0]), # 1 var
            ),
        ),
    ),
}


@attr.s(auto_attribs=True)
class CustomNoisyActuationSpec(ActuationSpec):
    # Struct to hold parameters for noise model
    robot: str = attr.ib(default="Universal")

    @robot.validator
    def check(self, attribute, value):
        assert value in custom_noise_models.keys(), f"{value} not a known robot"

    controller: str = attr.ib(default="Proportional")

    @controller.validator
    def check(self, attribute, value):
        assert value in [
            "Proportional"
        ], f"{value} not a known controller"

    noise_multiplier: float = 1.0


_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2


def _noisy_action_impl(
    scene_node: hsim.SceneNode,
    translate_amount: float,
    rotate_amount: float,
    multiplier: float,
    model: MotionNoiseModel,
    motion_type: str,
):
    """
    multiplier is the
    ROTATION STD NOISE in case of rotational motion type,
    MOTION STD NOISE in case of linear motion type
    """
    # Perform the action in the coordinate system of the node
    transform = scene_node.transformation
    move_ax = -transform[_Z_AXIS].xyz
    perp_ax = transform[_X_AXIS].xyz

    # if the action is rotational, introduce a rotation error but NOT a translation error
    if motion_type == "rotational":
        translation_noise=np.array([0., 0.], dtype=np.float32)
    else: # is linear
        translation_noise = ( (model.linear.sample() * 10.0) * multiplier )/10.0     
    # apply the noise along the 2 axis
    scene_node.translate_local(
        move_ax * (translate_amount + translation_noise[0])
        + perp_ax * (translation_noise[1])
    )
    if motion_type == "linear":
        # if the movement was straight, add a bit of noise to rotation
        rot_noise = 10.0 * translate_amount * model.rotation.sample()
    else:
        rot_noise = ( (model.rotation.sample() * 10.0) * multiplier )/10.0
    
    scene_node.rotate_y_local(mn.Deg(rotate_amount) + mn.Deg(rot_noise))
    scene_node.rotation = scene_node.rotation.normalized()


@registry.register_move_fn(body_action=True)
class CustomNoisyMoveBackward(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: CustomNoisyActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            -actuation_spec.amount,
            0.0,
            actuation_spec.noise_multiplier,
            custom_noise_models[actuation_spec.robot][actuation_spec.controller].linear_motion,
            "linear",
        )


@registry.register_move_fn(body_action=True)
class CustomNoisyMoveForward(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: CustomNoisyActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            actuation_spec.amount,
            0.0,
            actuation_spec.noise_multiplier,
            custom_noise_models[actuation_spec.robot][actuation_spec.controller].linear_motion,
            "linear",
        )


@registry.register_move_fn(body_action=True)
class CustomNoisyTurnLeft(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: CustomNoisyActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            0.0,
            actuation_spec.amount,
            actuation_spec.noise_multiplier,
            custom_noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].rotational_motion,
            "rotational",
        )


@registry.register_move_fn(body_action=True)
class CustomNoisyTurnRight(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: CustomNoisyActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            0.0,
            -actuation_spec.amount,
            actuation_spec.noise_multiplier,
            custom_noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].rotational_motion,
            "rotational",
        )
