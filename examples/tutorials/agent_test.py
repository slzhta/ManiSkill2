from mani_skill.agents import REGISTERED_AGENTS
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.scene import ManiSkillScene

import dacite
from mani_skill.utils.structs.types import Array, SimConfig

import sapien
import sapien.physx
import sapien.physx as physx

import numpy as np
import torch

if __name__ == "__main__":

    _scene: ManiSkillScene = None
    num_envs = 1
    device = torch.device("cuda")

    _default_sim_cfg = SimConfig()
    merged_gpu_sim_cfg = _default_sim_cfg.dict()
    sim_cfg = dacite.from_dict(data_class=SimConfig, data=merged_gpu_sim_cfg, config=dacite.Config(strict=True))

    _sim_freq = sim_cfg.sim_freq
    _control_freq = sim_cfg.control_freq
    _control_mode = "pd_ee_delta_pose"

    ############################################################################################################
    #                                           Set up Scene                                                   #
    ############################################################################################################
    physx.set_scene_config(**sim_cfg.scene_cfg.dict())
    physx.set_default_material(**sim_cfg.default_materials_cfg.dict())

    if sapien.physx.is_gpu_enabled():
        physx_system = sapien.physx.PhysxGpuSystem()
        # Create the scenes in a square grid
        sub_scenes = []
        scene_grid_length = int(np.ceil(np.sqrt(num_envs)))
        for scene_idx in range(num_envs):
            scene_x, scene_y = (
                scene_idx % scene_grid_length,
                scene_idx // scene_grid_length,
            )
            scene = sapien.Scene(
                systems=[physx_system, sapien.render.RenderSystem()]
            )
            scene.physx_system.set_scene_offset(
                scene,
                [
                    scene_x * sim_cfg.spacing,
                    scene_y * sim_cfg.spacing,
                    0,
                ],
            )
            sub_scenes.append(scene)
    else:
        physx_system = sapien.physx.PhysxCpuSystem()
        sub_scenes = [
            sapien.Scene([physx_system, sapien.render.RenderSystem()])
        ]
    # create a "global" scene object that users can work with that is linked with all other scenes created
    _scene = ManiSkillScene(sub_scenes, sim_cfg=sim_cfg, device=device)
    physx_system.timestep = 1.0 / _sim_freq

    ############################################################################################################
    #                                            Load Agent                                                    #
    ############################################################################################################
    robot_init_qpos_noise=0.02

    robot_uid = "xarm7_ability"
    agent_cls = REGISTERED_AGENTS[robot_uid].agent_cls
    agent: BaseAgent = agent_cls(_scene,
                                _control_freq,
                                _control_mode,
                                agent_idx=None,)
    
    print("CLS:", agent_cls)
    
    ############################################################################################################
    #                                            Load Scene                                                    #
    ############################################################################################################

    # table_scene = TableSceneBuilder(
    #         self, robot_init_qpos_noise=self.robot_init_qpos_noise
    #     )
