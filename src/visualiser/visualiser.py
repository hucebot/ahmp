import numpy as np

import pinocchio as pin

from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf


class TrajoptVisualiser:
    def __init__(self, robot_wrapper):
        self.vis = MeshcatVisualizer(
            robot_wrapper.model,
            robot_wrapper.collision_model,
            robot_wrapper.visual_model,
        )
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.i = 0
        self.meshcat_viewer = self.vis.viewer

        # Add floor
        floor_length = 20.0
        floor_width = 20.0
        floor_thickness = 0.01
        floor = g.Box([floor_length, floor_width, floor_thickness])
        floor_material = g.MeshPhongMaterial(color=0x444444, reflectivity=0.8)
        self.vis.viewer["floor"].set_object(floor, floor_material)
        floor_pose = tf.translation_matrix([0, 0, -floor_thickness / 2])
        self.vis.viewer["floor"].set_transform(floor_pose)

        # Black background
        self.vis.setBackgroundColor(col_top=[0.0, 0.0, 0.0], col_bot=[0.0, 0.0, 0.0])

    def display_multiple_instances(self, robot_wrapper, q, opacity=0.5):

        instance_name = f"robot_{self.i}"
        self.i += 1

        instance_vis = MeshcatVisualizer(
            robot_wrapper.model,
            robot_wrapper.collision_model,
            robot_wrapper.visual_model,
        )
        instance_vis.viewer = self.meshcat_viewer
        instance_vis.loadViewerModel(instance_name)

        # Display the instance
        robot_wrapper.fk_all(q)
        instance_vis.display(q)

    def display_robot_q(self, robot_wrapper, q):
        robot_wrapper.fk_all(q)
        self.vis.display(q)

    def load_terrain(self, terrain):
        self.vis.viewer["terrain"].delete()
        n_samples = 2 * terrain.rows
        x_range = np.linspace(terrain.min_x, terrain.max_x, 4 * terrain.rows)
        y_range = np.linspace(terrain.min_y, terrain.max_y, 4 * terrain.cols)
        box_x = np.abs((terrain.max_x - terrain.min_x) / terrain.rows)
        box_y = np.abs((terrain.max_y - terrain.min_y) / terrain.cols)

        # Show grid heights above 0.0
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                height = terrain.height(x, y)
                obj = g.Sphere(0.005)
                translation = [x, y, height]
                transform = tf.translation_matrix(translation)
                self.vis.viewer["terrain"][f"sample_{i}_{j}"].set_object(obj)
                self.vis.viewer["terrain"][f"sample_{i}_{j}"].set_transform(transform)

    def load_vault_obstacle(self, z, x_min, x_max):
        self.vis.viewer["terrain"].delete()
        length = x_max - x_min
        box = g.Box([x_max - x_min, 1.0, z])
        translation = [x_min + (length / 2), 0.0, z / 2]
        transform = tf.translation_matrix(translation)
        self.vis.viewer["terrain"].set_object(box)
        self.vis.viewer["terrain"].set_transform(transform)

    def load_chimney_walls(self, y):
        self.vis.viewer["terrain"].delete()
        wall = g.Box([1.5, 0.2, 100])
        translation_l = [0.0, y + (0.2 / 2), 100 / 2]
        translation_r = [0.0, -y - (0.2 / 2), 100 / 2]
        transform_l = tf.translation_matrix(translation_l)
        transform_r = tf.translation_matrix(translation_r)
        self.vis.viewer["terrain"]["wall_left"].set_object(wall)
        self.vis.viewer["terrain"]["wall_left"].set_transform(transform_l)
        self.vis.viewer["terrain"]["wall_right"].set_object(wall)
        self.vis.viewer["terrain"]["wall_right"].set_transform(transform_r)

    def load_stairs(self, x_start, dx, dz):
        self.vis.viewer["terrain"].delete()
        n_stairs = int(2 // dx)
        for k in range(n_stairs):
            stair = g.Box([dx, 1.0, (k + 1) * dz])
            translation = [k * dx + (dx / 2) + x_start, 0.0, (k + 1) * dz / 2]
            transform = tf.translation_matrix(translation)
            self.vis.viewer["terrain"][f"stair_{k}"].set_object(stair)
            self.vis.viewer["terrain"][f"stair_{k}"].set_transform(transform)

    def load_handrails(self, rail_y, rail_z, rail_w):
        self.vis.viewer["terrain"].delete()
        rail_l = 3.3
        rail_left = g.Box([rail_l, rail_w, rail_w])
        rail_right = g.Box([rail_l, rail_w, rail_w])
        translation_left = [rail_l / 2, rail_y, rail_z]
        translation_right = [rail_l / 2, -rail_y, rail_z]
        transform_left = tf.translation_matrix(translation_left)
        transform_right = tf.translation_matrix(translation_right)
        self.vis.viewer["terrain"]["rail_left"].set_object(rail_left)
        self.vis.viewer["terrain"]["rail_right"].set_object(rail_right)
        self.vis.viewer["terrain"]["rail_left"].set_transform(transform_left)
        self.vis.viewer["terrain"]["rail_right"].set_transform(transform_right)

    def update_forces(self, robot_wrapper, forces_dict, scale=1):
        self.vis.viewer["forces"].delete()
        for fid, f_F in forces_dict.items():
            # Convert force vector to float (to avoid dtype issues)
            force_arrow = scale * f_F

            fid = robot_wrapper.model.getFrameId(fid)

            # Get world position of the frame
            oMF = robot_wrapper.data.oMf[fid]
            force_start = oMF.translation  # Origin of force
            force_end = force_start + oMF.rotation @ force_arrow  # Force direction

            # Create vertices for the line
            verts = np.vstack((force_start, force_end)).T

            # Create the line geometry
            line_geometry = g.Line(
                g.PointsGeometry(verts),
                g.LineBasicMaterial(color=0xFF0000, linewidth=5),
            )

            # Set the force geometry in the visualizer
            self.vis.viewer["forces"][f"force_{fid}"].set_object(line_geometry)
