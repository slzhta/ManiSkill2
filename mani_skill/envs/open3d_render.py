import numpy as np
from typing import cast, Any, Optional
from transforms3d.euler import euler2quat

from sapien import Entity
#from sapien.physx import PhysxRigidBaseComponent
from sapien.render import RenderBodyComponent, RenderShape, RenderShapeBox, RenderShapeTriangleMesh, RenderShapeTriangleMeshPart, RenderShapeCylinder
from sapien import Pose

from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxCollisionShape,
    PhysxCollisionShapeBox,
    PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh,
    PhysxCollisionShapeCylinder,
    PhysxCollisionShapePlane,
    PhysxCollisionShapeSphere,
    PhysxCollisionShapeTriangleMesh,
    PhysxRigidBaseComponent,
)


import open3d as o3d
import open3d.visualization.rendering as rendering

def shape2render_obj(shape: RenderShape) -> o3d.geometry.Geometry3D:
    # shape = shapes[0]
    #if isinstance(component, PhysxArticulationLinkComponent):  # articulation link
    #    pose = shape.local_pose
    #else:
    #    pose = component.entity.pose * shape.local_pose
    if isinstance(shape, RenderShapeBox):
        #collision_geom = Box(side=shape.half_size * 2)
        o3d_shape =  o3d.geometry.TriangleMesh.create_box(*shape.half_size * 2)
    elif isinstance(shape, RenderShapeCylinder):
        o3d_shape = o3d.geometry.TriangleMesh.create_cylinder(radius=shape.radius, height=shape.half_length * 2)
    elif isinstance(shape, RenderShapeTriangleMesh):
        # assert len(shape.parts) == 1, "Only support single part for now"
        vertices = []
        indexes = []
        total = 0
        for k in shape.parts:
            vertices.append(k.vertices)
            indexes.append(k.triangles + total)
            total += len(k.vertices)
        o3d_shape =  o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.concatenate(vertices)), triangles=o3d.utility.Vector3iVector(np.concatenate(indexes)))
    else: 
        raise NotImplementedError(f"Shape {shape} not supported")
    return o3d_shape


class OurRenderShape:
    def __init__(self, shape: RenderShape, local_pose: Pose, o3d_shape: Any, entity: Optional[Entity] = None) -> None:
        self.shape = shape
        self.local_pose = local_pose
        self.o3d_shape = o3d_shape
        self.entity = entity

class CustomizedViewer:
    def __init__(self) -> None:
        self.render_shapes: dict[str, list[OurRenderShape]] = {}

    def add_entity(self, entity: Entity):
        component = entity.find_component_by_type(RenderBodyComponent)
        if component is None:
            return
        assert (
            component is not None
        ), f"No RenderBodhyComponent found in {entity.name}: {entity.components=}"

        component = cast(RenderBodyComponent, component)
        shapes = component.render_shapes

        if entity.name in self.render_shapes:
            assert len(shapes) == len(self.render_shapes[entity.name]), "Number of shapes mismatch"
            for shape, cur_col_shape in zip(shapes, self.render_shapes[entity.name]):
                cur_col_shape.entity = entity
                cur_col_shape.shape = shape
            return
        
        assert len(shapes) > 0, "No collision shapes found in entity"

        col_shape: list[OurRenderShape] = []
        for idx, shape in enumerate(shapes):
            shape.local_pose
            o3d_obj = shape2render_obj(shape)
            col_shape.append(OurRenderShape(shape, shape.local_pose, o3d_obj, entity))
        self.render_shapes[entity.name] = col_shape

    def update_entity(self, name: str, pose: Optional[Pose]=None):
        for shape in self.render_shapes[name]:
            pose = pose or (shape.entity.get_pose() if shape.entity else None)
            # shape.entity.set_pose(pose)
            assert pose is not None, f"Pose is None for {name}"
            pose = pose * shape.local_pose
            # shape.local_pose = pose
            shape.o3d_shape.transform(pose.to_transformation_matrix())

    def update_world(self):
        for name in self.render_shapes:
            self.update_entity(name)

    def update_all(self):
        for name in self.render_shapes:
            self.update_entity(name)

    def view_all_by_figure(self, filename):
        vis = o3d.visualization.Visualizer()

        vis.create_window()

        for name in self.render_shapes:
            for shape in self.render_shapes[name]:
                vis.add_geometry(shape.o3d_shape)
        
        camera_target = [0, 0, 0]

        vis.get_view_control().set_lookat(camera_target)
        vis.get_view_control().camera_local_translate(-0.3, 0.3, -0.3)
        vis.get_view_control().set_front([0.3, -0.3, 0.3])
        vis.get_view_control().set_up([-1, 1, 0])

        vis.capture_screen_image(filename, do_render=True)
        vis.destroy_window()
    
    def view_all_with_np(self):
        vis = o3d.visualization.Visualizer()

        vis.create_window()

        for name in self.render_shapes:
            for shape in self.render_shapes[name]:
                vis.add_geometry(shape.o3d_shape)
        
        camera_target = [0, 0, 0]

        vis.get_view_control().set_lookat(camera_target)
        vis.get_view_control().camera_local_translate(-0.3, 0.3, -0.3)
        vis.get_view_control().set_front([0.3, -0.3, 0.3])
        vis.get_view_control().set_up([-1, 1, 0])

        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.array(image)

        vis.destroy_window()

        return image_np

    def view_all_with_np_new(self):

        vis = o3d.visualization.Visualizer()

        width = 1920
        height = 1080
        vis.create_window(width=width, height=height)

        for name in self.render_shapes:
            for shape in self.render_shapes[name]:
                vis.add_geometry(shape.o3d_shape)
        
        # camera_target = [0, 0, 0]
        vis_ctrl = vis.get_view_control()
        # camera_parameter = vis_ctrl.convert_to_pinhole_camera_parameters()

        # ex = camera_parameter.extrinsic

        # x_move = ex[2][3] - 1.1160079
        # y_move = ex[0][3] + 0.3
        # z_move = ex[1][3] - 0.3

        vis_ctrl.set_front([-1, 0, 0])
        vis_ctrl.set_up([0, 0, 1])

        
        # vis_ctrl.set_lookat([0.6, -0.6, 0.6])
        # vis_ctrl.set_front([-0.6, 0.6, -0.6])
        vis_ctrl.set_lookat([0, 0, 0])

        vis_ctrl.camera_local_translate(-0.5, 0.2, 0)

        # vis_ctrl.set_up([-1, 1, 0])

        # sec_camera_parameter = vis_ctrl.convert_to_pinhole_camera_parameters()
        # print("sec:", sec_camera_parameter.extrinsic)

        # vis_ctrl.set_front([0.3, -0.3, 0.3])
        # vis_ctrl.set_up([-1, 1, 0])

        # new_camera_parameter = vis_ctrl.convert_to_pinhole_camera_parameters()
        # print("new:", new_camera_parameter.extrinsic)
        # camera_parameter.extrinsic = np.array([[ 0.70710678,  0.70710678,  0.        , -0.        ],
        #                                        [ 0.70710678, -0.70710678, -0.        , -1.09053173],
        #                                        [-0.57735027,  0.57735027, -0.57735027,  1.33562314],
        #                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        
        # print("width:", camera_parameter.intrinsic.width)
        # print("height :", camera_parameter.intrinsic.height)
        # print("matrix:", camera_parameter.intrinsic.intrinsic_matrix)

        # camera_parameter.intrinsic.set_intrinsics(width, height, focal * width, focal * width, width / 2 - 0.5, height / 2 - 0.5)

        # vis_ctrl.convert_from_pinhole_camera_parameters(camera_parameter)
        # print(camera_parameter.extrinsic)

        # vis.get_view_control().set_lookat(camera_target)
        # vis.get_view_control().camera_local_translate(-0.3, 0.3, -0.3)
        # vis.get_view_control().set_front([0.3, -0.3, 0.3])
        # vis.get_view_control().set_up([-1, 1, 0])

        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.array(image)

        vis.destroy_window()

        return image_np
            

if __name__ == "__main__":
    print("Start runing")
    import sapien
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    print("Finish set renderer")

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    # scene.load_widget_from_package("demo_arena", "DemoArena")

    print("Finish set timestep")

    material = renderer.create_material()
    material.base_color = [0.5, 0.5, 0.5, 1]
    scene.add_ground(-1, render_material=material)

    print("Finish add ground")

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    print("Finish add light")

    loader = scene.create_urdf_loader()
    loader.multiple_collisions_decomposition = "coacd"
    loader.load_multiple_collisions_from_file = True
    loader.fix_root_link = True
    robot = loader.load('../assets/robots/xarm7/xarm7_five_finger_right_hand.urdf')
    robot.name = 'robot'

    print("Finish load robot")

    viwer = CustomizedViewer()

    print("Finish create viewer")

    robot.set_qpos(np.zeros(robot.dof))
    for link in robot.get_links():
        viwer.add_entity(link.entity)

    print("Finish set entity")

    viwer.update_all()

    print("Update all")

    viwer.view_all_by_figure("./test.png")

    print("Finish")