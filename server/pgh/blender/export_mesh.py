"""Clean, orient, scale and export a reconstructed mesh. Runs inside Blender.

Invoked as ``blender -b --factory-startup -P export_mesh.py -- <job.json>``. It runs in
Blender's own interpreter, so it can import neither ``pgh`` nor numpy and must not try:
everything it needs arrives in the job file, and every decision needing real reasoning
was made in ``orient.py``, where it could be tested without a 300 MB application.

What it does, in order:

1. Import the mesh and **merge vertices by distance**. This is not tidying: a textured
   GLB splits its vertices at every texture-atlas seam, so the cup mesh arrives as
   11,015 apparently disconnected islands against a real 7 (HANDOVER 6.23). Merging
   heals them exactly -- 337,800 vertices back to the 249,022 the PLY has -- and UVs
   survive it because Blender stores them per face-corner rather than per vertex.
   Without this step, "keep the largest connected component" keeps half a percent of the
   model and throws the subject away.
2. Keep the largest connected shell, if asked, dropping the floating fragments that a
   reflective or transparent surface leaves scattered around a reconstruction.
3. Orient, scale, then centre -- in that order, since each depends on the last. The up
   axis arrives in the reconstruction's coordinates and is carried through the object's
   own transform before the rotation is built, because the importer has already applied
   an axis convention of its own.
4. Decimate to a face budget, if one was given.
5. Export each requested format, then render a turntable.

It reports back as one ``PGH_RESULT <json>`` line on stdout. Blender writes a great deal
to stdout that is not a result, hence the prefix.
"""

import json
import math
import os
import sys

import bpy
import mathutils

RESULT_PREFIX = "PGH_RESULT "

#: Merge threshold as a fraction of the bounding-box diagonal. Atlas seams are exact
#: duplicates, so this only has to beat float round-trip through glTF; relative rather
#: than absolute so it means the same thing whatever units the reconstruction is in.
MERGE_RATIO = 2e-6


def log(message):
    print("PGH " + str(message), flush=True)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


#: glTF is a Y-up format and Blender is Z-up, so its importer rewrites every vertex as
#: (x, -z, y) on the way in. It does this to the mesh data rather than to the object
#: transform, so there is nothing at runtime to read it back off -- the object arrives
#: with an identity matrix and coordinates that are no longer the reconstruction's.
#: This is the inverse, applied on import so that everything downstream, the up axis
#: included, is speaking one coordinate system. The glTF *exporter* re-applies the
#: conversion on the way out, so the round trip is symmetric.
GLTF_TO_RECONSTRUCTION = mathutils.Matrix(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
).to_4x4()


def import_mesh(path):
    """Import by extension, in the reconstruction's own coordinates.

    Every importer here is either told not to convert axes or has its conversion undone
    afterwards. Without that, an up axis computed from the camera poses is applied in a
    frame 90 degrees away from the one the mesh landed in -- which exports the model
    standing on its edge while every number in the report looks reasonable.
    """
    lower = path.lower()
    converted = False
    if lower.endswith(".glb") or lower.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
        converted = True
    elif lower.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif lower.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:
        raise RuntimeError("cannot import %s: unknown extension" % path)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("%s imported with no mesh in it" % path)
    if len(meshes) > 1:
        # A glTF scene can arrive as several nodes; joining leaves one thing to transform.
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        meshes = [meshes[0]]

    obj = meshes[0]
    if converted:
        obj.matrix_world = GLTF_TO_RECONSTRUCTION @ obj.matrix_world
        bpy.context.view_layer.update()
    return obj


def bounds(obj):
    """World-space min and max corners of the object's bounding box."""
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    low = mathutils.Vector(
        (min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners))
    )
    high = mathutils.Vector(
        (max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners))
    )
    return low, high


def heal_seams(obj):
    """Merge the duplicate vertices a texture atlas leaves behind. See the module note."""
    low, high = bounds(obj)
    threshold = max((high - low).length * MERGE_RATIO, 1e-9)
    before = len(obj.data.vertices)

    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode="OBJECT")

    log("healed %d split vertices at threshold %.3g" % (before - len(obj.data.vertices), threshold))
    return before - len(obj.data.vertices)


def largest_component(obj):
    """Keep the biggest connected shell. Returns (object, components, faces dropped)."""
    before = len(obj.data.polygons)
    activate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = [o for o in bpy.data.objects if o.type == "MESH"]
    parts.sort(key=lambda o: len(o.data.polygons), reverse=True)
    keep = parts[0]
    for part in parts[1:]:
        bpy.data.objects.remove(part, do_unlink=True)

    dropped = before - len(keep.data.polygons)
    log("components=%d kept=%d dropped=%d" % (len(parts), len(keep.data.polygons), dropped))
    return keep, len(parts), dropped


def resolve_scale(obj, scale, fit):
    """The factor to scale by: either given outright, or derived from a measurement.

    A measured dimension can only become a factor once the model is standing up, because
    "height" means nothing until then -- so this runs after the orientation is applied
    and reads the bounding box it produced.
    """
    if scale:
        return float(scale)
    if not fit or not fit.get("mm"):
        return 1.0

    low, high = bounds(obj)
    size = high - low
    axis = fit.get("axis", "height")
    current = {
        "width": size.x,
        "depth": size.y,
        "height": size.z,
        "longest": max(size.x, size.y, size.z),
    }.get(axis, size.z)

    if current <= 1e-9:
        raise RuntimeError(
            "the model measures nothing along %s, so a real dimension cannot be "
            "turned into a scale" % axis
        )
    factor = float(fit["mm"]) / float(current)
    log("scale %.6f mm per unit, from %s = %s mm" % (factor, axis, fit["mm"]))
    return factor


def orient_to_z(obj, axis):
    """Turn the model so the reconstruction's up axis points along Blender's +Z.

    The axis arrives in the reconstruction's coordinates, and ``import_mesh`` has already
    put the object's *world* coordinates into that same frame -- see
    GLTF_TO_RECONSTRUCTION for why that is not automatic. So the axis is used as it
    stands. It must not be carried through ``matrix_world`` on the way: that maps the
    mesh's own data coordinates into the world, and the axis is not in data coordinates.
    Doing it anyway applies the glTF conversion a second time, which is a 90 degree error
    that leaves the model standing on its edge with every number in the report looking
    entirely reasonable.
    """
    if axis is None:
        return
    world_axis = mathutils.Vector(axis).normalized()
    rotation = world_axis.rotation_difference(mathutils.Vector((0.0, 0.0, 1.0)))
    obj.matrix_world = rotation.to_matrix().to_4x4() @ obj.matrix_world
    bpy.context.view_layer.update()
    log("up axis %s taken to +Z" % (tuple(round(v, 3) for v in world_axis),))


def apply_transform(obj, axis, scale, fit, centre):
    """Orient, scale, then centre -- in that order, because each depends on the last.

    Centring is last because the bounding box has to be measured in the frame the model
    ends up in; done first, the model is centred on the box it used to have. It sits the
    model on z=0 rather than centring it vertically, which is what a slicer expects.
    """
    orient_to_z(obj, axis)

    scale = resolve_scale(obj, scale, fit)
    if scale and scale != 1.0:
        obj.matrix_world = mathutils.Matrix.Scale(scale, 4) @ obj.matrix_world
        bpy.context.view_layer.update()

    if centre:
        low, high = bounds(obj)
        middle = (low + high) / 2.0
        obj.matrix_world = (
            mathutils.Matrix.Translation(mathutils.Vector((-middle.x, -middle.y, -low.z)))
            @ obj.matrix_world
        )
        bpy.context.view_layer.update()

    return scale


def decimate(obj, target_faces):
    if not target_faces or len(obj.data.polygons) <= target_faces:
        return len(obj.data.polygons)
    activate(obj)
    modifier = obj.modifiers.new(name="decimate", type="DECIMATE")
    modifier.ratio = max(0.001, float(target_faces) / len(obj.data.polygons))
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    log("decimated to %d faces" % len(obj.data.polygons))
    return len(obj.data.polygons)


def base_colour_image(obj):
    """The texture this mesh is painted with, wherever the importer parked it."""
    for slot in obj.material_slots:
        material = slot.material
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                return node.image
    return None


def rebuild_material_for_obj(obj, directory):
    """Give the mesh a plain Principled BSDF with the atlas on Base Color.

    OpenMVS marks its glTF material KHR_materials_unlit (HANDOVER 6.26), which the
    importer turns into a node graph built around a Background/Emission-style shader.
    The OBJ exporter only knows how to write a texture that reaches Base Color of a
    Principled BSDF, so it exported the material as a flat grey Kd with no map_Kd at
    all -- an untextured OBJ, from a mesh whose texture was sitting right there.

    The image also arrives *packed* into the .blend with no path on disk, so even a
    recognised material would have written a map_Kd pointing at a file that does not
    exist. It has to be unpacked before path_mode='COPY' can put it beside the .obj.
    """
    image = base_colour_image(obj)
    if image is None:
        return None

    if image.packed_file is not None:
        image.filepath_raw = os.path.join(directory, "%s_texture.png" % image.name)
        image.file_format = "PNG"
        image.save()
        image.unpack(method="REMOVE")
        image.filepath = image.filepath_raw

    material = bpy.data.materials.new("pgh_export")
    material.use_nodes = True
    tree = material.node_tree
    principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    tree.links.new(principled.inputs["Base Color"], texture.outputs["Color"])

    obj.data.materials.clear()
    obj.data.materials.append(material)
    log("rebuilt the material as a Principled BSDF for OBJ export")
    return image


def export(obj, directory, stem, formats):
    """Write each requested format. Returns (models, sidecars).

    An OBJ is three files, not one, and the other two are useless on their own --
    a downloaded .obj without its .mtl and its texture is an untextured mesh. They
    are reported separately from the models so the stage can record them all and the
    page can offer them without pretending the .mtl is a model in its own right.
    """
    activate(obj)
    written = {}
    sidecars = {}
    if formats.get("glb"):
        path = os.path.join(directory, stem + ".glb")
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)
        written["glb"] = os.path.basename(path)
    if formats.get("obj"):
        path = os.path.join(directory, stem + ".obj")
        # GLB last, OBJ second: rebuilding the material for OBJ must not change what
        # the GLB was written from.
        rebuild_material_for_obj(obj, directory)
        bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            export_materials=True,
            path_mode="COPY",
            # The importer is told Y/Z so it converts nothing; the exporter defaults
            # to NEGATIVE_Z/Y and would turn the OBJ ninety degrees relative to the
            # GLB and STL written from the same mesh in the same run. Same class of
            # error as 6.25, and just as invisible in the report.
            forward_axis="Y",
            up_axis="Z",
        )
        written["obj"] = os.path.basename(path)
        # Discover what came with it rather than predicting the names: the exporter
        # chooses the .mtl name, and path_mode='COPY' chooses the texture's.
        for name in sorted(os.listdir(directory)):
            lower = name.lower()
            if name == written["obj"]:
                continue
            if lower.endswith(".mtl"):
                sidecars["obj_material"] = name
            elif lower.endswith((".png", ".jpg", ".jpeg")) and stem in name:
                sidecars["obj_texture"] = name
    if formats.get("stl"):
        path = os.path.join(directory, stem + ".stl")
        # STL carries geometry only: no colour, no units, no way to say what size it is.
        # That is why scale has to be right before this line rather than after it.
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
        written["stl"] = os.path.basename(path)
    log("exported " + ", ".join(sorted(written)))
    if sidecars:
        log("sidecars: " + ", ".join("%s=%s" % kv for kv in sorted(sidecars.items())))
    return written, sidecars


def turntable(obj, directory, frames, resolution):
    """Orbit a camera once around the model, for the record."""
    if not frames:
        return 0
    os.makedirs(directory, exist_ok=True)

    scene = bpy.context.scene
    # EEVEE is named differently across Blender versions and an unknown enum raises, so
    # take whichever real-time engine this build actually offers.
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if candidate in engines:
            scene.render.engine = candidate
            break
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("turntable")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world

    light_data = bpy.data.lights.new("key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    light.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    scene.collection.objects.link(light)

    low, high = bounds(obj)
    centre = (low + high) / 2.0
    radius = max((high - low).length, 1e-6) * 1.4

    # Aim at an empty placed on the geometry, not at the object. TRACK_TO points at the
    # target's ORIGIN, and after the transforms above the mesh's origin can be a long way
    # from the mesh -- which frames the model small and off to one side.
    focus = bpy.data.objects.new("focus", None)
    focus.location = centre
    scene.collection.objects.link(focus)

    camera_data = bpy.data.cameras.new("turntable")
    camera = bpy.data.objects.new("turntable", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new(type="TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    for index in range(frames):
        angle = 2 * math.pi * index / frames
        camera.location = (
            centre.x + radius * math.cos(angle),
            centre.y + radius * math.sin(angle),
            centre.z + radius * 0.35,
        )
        scene.render.filepath = os.path.join(directory, "frame_%03d.png" % index)
        bpy.ops.render.render(write_still=True)
    log("rendered %d turntable frames" % frames)
    return frames


def main():
    with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as handle:
        job = json.load(handle)

    clear_scene()
    obj = import_mesh(job["mesh"])
    faces_in = len(obj.data.polygons)

    healed = heal_seams(obj)
    components = 1
    dropped = 0
    if job.get("largest_component_only"):
        obj, components, dropped = largest_component(obj)

    scale = apply_transform(
        obj, job.get("up_axis"), job.get("scale"), job.get("fit"), job.get("centre", True)
    )
    faces_out = decimate(obj, job.get("target_faces"))

    low, high = bounds(obj)
    size = high - low

    written, sidecars = export(
        obj, job["out_dir"], job.get("stem", "model"), job.get("formats", {})
    )
    frames = turntable(
        obj,
        os.path.join(job["out_dir"], "turntable"),
        job.get("turntable_frames", 0),
        job.get("turntable_resolution", 720),
    )

    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "faces_in": faces_in,
                "faces_out": faces_out,
                "scale": scale,
                "vertices_healed": healed,
                "components": components,
                "dropped_faces": dropped,
                "dimensions": [round(size.x, 6), round(size.y, 6), round(size.z, 6)],
                "files": written,
                "sidecars": sidecars,
                "turntable_frames": frames,
            }
        ),
        flush=True,
    )


main()
