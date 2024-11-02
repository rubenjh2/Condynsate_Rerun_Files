# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""Example of an executable data-loader plugin for the Rerun Viewer for URDF files."""
from __future__ import annotations

import os
import pathlib
from typing import Optional

import numpy as np
import rerun as rr  # pip install rerun-sdk
import scipy.spatial.transform as st
import trimesh
import trimesh.visual
from PIL import Image
from urdf_parser_py import urdf as urdf_parser

from pathlib import Path


class URDFLogger:
    """Class to log a URDF to Rerun."""

    def __init__(self, filepath: str, entity_path_prefix: Optional[str]) -> None:
        self.urdf = urdf_parser.URDF.from_xml_file(filepath)
        self.entity_path_prefix = entity_path_prefix
        self.mat_name_to_mat = {mat.name: mat for mat in self.urdf.materials}

    def link_entity_path(self, link: urdf_parser.Link) -> str:
        """Return the entity path for the URDF link."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, link.name)[0::2]  # skip the joints
        return self.add_entity_path_prefix("/".join(link_names))

    def joint_entity_path(self, joint: urdf_parser.Joint) -> str:
        """Return the entity path for the URDF joint."""
        root_name = self.urdf.get_root()
        joint_names = self.urdf.get_chain(root_name, joint.child)[0::2]  # skip the links
        return self.add_entity_path_prefix("/".join(joint_names))

    def add_entity_path_prefix(self, entity_path: str) -> str:
        """Add prefix (if passed) to entity path."""
        if self.entity_path_prefix is not None:
            return f"{self.entity_path_prefix}/{entity_path}"
        return entity_path

    def log(self) -> None:
        """Log a URDF file to Rerun."""
        for joint in self.urdf.joints:
            entity_path = self.joint_entity_path(joint)
            self.log_joint(entity_path, joint)

        for link in self.urdf.links:
            entity_path = self.link_entity_path(link)
            self.log_link(entity_path, link)

    def log_link(self, entity_path: str, link: urdf_parser.Link) -> None:
        """Log a URDF link to Rerun."""
        # create one mesh out of all visuals
        for i, visual in enumerate(link.visuals):
            self.log_visual(entity_path + f"/visual_{i}", visual)

    def log_joint(self, entity_path: str, joint: urdf_parser.Joint) -> None:
        """Log a URDF joint to Rerun."""
        translation = rotation = None

        if joint.origin is not None and joint.origin.xyz is not None:
            translation = joint.origin.xyz

        if joint.origin is not None and joint.origin.rpy is not None:
            rotation = st.Rotation.from_euler("xyz", joint.origin.rpy).as_matrix()

        rr.log(entity_path, rr.Transform3D(translation=translation, mat3x3=rotation))

    def log_visual(self, entity_path: str, visual: urdf_parser.Visual) -> None:
        """Log a URDF visual to Rerun."""
        material = None
        if visual.material is not None:
            if visual.material.color is None and visual.material.texture is None:
                # use globally defined material
                material = self.mat_name_to_mat[visual.material.name]
            else:
                material = visual.material

        transform = np.eye(4)
        if visual.origin is not None and visual.origin.xyz is not None:
            transform[:3, 3] = visual.origin.xyz
        if visual.origin is not None and visual.origin.rpy is not None:
            transform[:3, :3] = st.Rotation.from_euler("xyz", visual.origin.rpy).as_matrix()

        if isinstance(visual.geometry, urdf_parser.Mesh):
            resolved_path = resolve_path(visual.geometry.filename)
            mesh_scale = visual.geometry.scale
            mesh_or_scene = trimesh.load_mesh(resolved_path)
            if mesh_scale is not None:
                transform[:3, :3] *= mesh_scale
        elif isinstance(visual.geometry, urdf_parser.Box):
            mesh_or_scene = trimesh.creation.box(extents=visual.geometry.size)
        elif isinstance(visual.geometry, urdf_parser.Cylinder):
            mesh_or_scene = trimesh.creation.cylinder(
                radius=visual.geometry.radius,
                height=visual.geometry.length,
            )
        elif isinstance(visual.geometry, urdf_parser.Sphere):
            mesh_or_scene = trimesh.creation.icosphere(
                radius=visual.geometry.radius,
            )
        else:
            rr.log(
                "",
                rr.TextLog("Unsupported geometry type: " + str(type(visual.geometry))),
            )
            mesh_or_scene = trimesh.Trimesh()

        mesh_or_scene.apply_transform(transform)

        if isinstance(mesh_or_scene, trimesh.Scene):
            scene = mesh_or_scene
            # use dump to apply scene graph transforms and get a list of transformed meshes
            for i, mesh in enumerate(scene_to_trimeshes(scene)):
                if material is not None and not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                    if material.color is not None:
                        mesh.visual = trimesh.visual.ColorVisuals()
                        mesh.visual.vertex_colors = material.color.rgba
                    elif material.texture is not None:
                        texture_path = resolve_path(material.texture.filename)
                        mesh.visual = trimesh.visual.texture.TextureVisuals(image=Image.open(texture_path))
                log_trimesh(entity_path + f"/{i}", mesh)
        else:
            mesh = mesh_or_scene
            if material is not None and not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                if material.color is not None:
                    mesh.visual = trimesh.visual.ColorVisuals()
                    mesh.visual.vertex_colors = material.color.rgba
                elif material.texture is not None:
                    texture_path = resolve_path(material.texture.filename)
                    mesh.visual = trimesh.visual.texture.TextureVisuals(image=Image.open(texture_path))
            log_trimesh(entity_path, mesh)


def scene_to_trimeshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    """
    Convert a trimesh.Scene to a list of trimesh.Trimesh.

    Skips objects that are not an instance of trimesh.Trimesh.
    """
    trimeshes = []
    scene_dump = scene.dump()
    geometries = [scene_dump] if not isinstance(scene_dump, list) else scene_dump
    for geometry in geometries:
        if isinstance(geometry, trimesh.Trimesh):
            trimeshes.append(geometry)
        elif isinstance(geometry, trimesh.Scene):
            trimeshes.extend(scene_to_trimeshes(geometry))
    return trimeshes


def log_trimesh(entity_path: str, mesh: trimesh.Trimesh) -> None:
    vertex_colors = albedo_texture = vertex_texcoords = None

    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        vertex_colors = mesh.visual.vertex_colors
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        trimesh_material = mesh.visual.material

        if mesh.visual.uv is not None:
            vertex_texcoords = mesh.visual.uv
            # Trimesh uses the OpenGL convention for UV coordinates, so we need to flip the V coordinate
            # since Rerun uses the Vulkan/Metal/DX12/WebGPU convention.
            vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

        if isinstance(trimesh_material, trimesh.visual.material.PBRMaterial):
            if trimesh_material.baseColorTexture is not None:
                albedo_texture = pil_image_to_albedo_texture(trimesh_material.baseColorTexture)
            elif trimesh_material.baseColorFactor is not None:
                vertex_colors = trimesh_material.baseColorFactor
        elif isinstance(trimesh_material, trimesh.visual.material.SimpleMaterial):
            if trimesh_material.image is not None:
                albedo_texture = pil_image_to_albedo_texture(trimesh_material.image)
            else:
                vertex_colors = mesh.visual.to_color().vertex_colors

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=vertex_colors,
            albedo_texture=albedo_texture,
            vertex_texcoords=vertex_texcoords,
        ),
        timeless=True,
    )

#-------------------

def resolve_path(path_str: str) -> str: #All URDF and stl files must be in "vis" folder
    
    if not "vis/" in path_str:
        path_str = os.path.join("vis", path_str)
        
    if os.path.isfile(path_str):
        return path_str
    else:
        # Resolve relative paths based on the current directory
        return str(Path(path_str).resolve())

#-------------------


#NEXT REMOVE ALL COMMAND LINE ARGUMENTS AND REPLACE DEPENDENT LOGIC WITH DIRECT VALUES LIKE PATH PREFIX

def main() -> None:
    # The Rerun Viewer will always pass these two pieces of information:
    # 1. The path to be loaded, as a positional arg.
    # 2. A shared recording ID, via the `--recording-id` flag.
    #
    # It is up to you whether you make use of that shared recording ID or not.
    # If you use it, the data will end up in the same recording as all other plugins interested in
    # that file, otherwise you can just create a dedicated recording for it. Or both.
    return None
