"""Build a synthetic two-camera turntable capture with ground truth. Runs inside Blender.

The harness has never been exercised on the capture mode it was written for. Every run on
record is ``camera_orbits`` -- a handheld camera circling a still subject -- and the one
masked run among them (``cup-1``) shows masking making things *worse*, because there the
background is rigid with the subject and supplies the features that close the orbit.

The face scan goes the other way: the cameras are bolted down and the subject spins on a
chair. The room is then static in the world while the subject is not, and SfM -- which
solves for camera motion across a rigid scene -- has an exact, attractive, wrong answer
available to it: that nothing moved. Masking is what takes that answer away. This scene
exists to prove that it does, and it is the only capture we can build where the right
answer is known in advance.

Invoked either way::

    blender -b --factory-startup -P make_suzanne_turntable.py -- job.json
    blender -b --factory-startup -P make_suzanne_turntable.py -- job.json --render

or, in a live session, ``exec(open(path).read())`` then ``build(CONFIG)`` -- which is worth
doing first, because framing is the one thing cheaper to check by eye than to assert.

It runs in Blender's own interpreter, so like ``pgh/blender/export_mesh.py`` it can import
neither ``pgh`` nor numpy and must not try. It reports one ``PGH_RESULT <json>`` line on
stdout; Blender writes a great deal to stdout that is not a result, hence the prefix.

What comes out, beside the frames:

* ``suzanne_gt.ply``   -- the true surface, subdivision applied, at render scale.
* ``cameras.json``     -- true intrinsics, and each camera's pose *in the subject's frame*,
  which is the thing COLMAP actually reconstructs. The cameras never move in the world, so
  their literal transforms say nothing at all.
* ``rig.json``         -- the lens-to-lens baseline, **measured from the two cameras' world
  matrices** rather than read back off the parameters that placed them. The capture
  checklist asks for a measured baseline; measuring it the same way twice is not a check.
* ``scene.json``       -- true subject height, degrees per frame, fps, resolution.
"""

import json
import math
import os
import sys

import bpy
import mathutils

RESULT_PREFIX = "PGH_RESULT "

#: Frames advance the subject by one degree. 600 of them is 1.667 revolutions, which leaves
#: better than the checklist's 1.5-revolution minimum still standing after the two clips are
#: trimmed to their overlapping window by the deliberate start offset.
CONFIG = {
    "out_dir": r"D:\pgh-test\suzanne",
    "frames": 600,
    "degrees_per_frame": 1.0,
    "fps": 30,
    #: 2560 is exactly ExtractParams.max_dimension, so the pipeline's own downscale is a
    #: no-op and cannot confound a comparison against ground truth.
    "resolution": [1440, 2560],
    "subject_height_m": 0.600,
    "subject_centre_m": [0.0, 0.0, 1.200],
    #: Left as None the standoff is computed from the subject's actual swept radius, which
    #: is the only way to be sure nothing leaves the frame. Suzanne is wider than she is
    #: tall -- 834 mm ear to ear against a 600 mm height -- and a guessed distance clips the
    #: ears at exactly the rotations where they matter most.
    "camera_distance_m": None,
    #: Fraction of the frame's short side the subject's swept circle should fill.
    "target_fill": 0.72,
    "lens_mm": 35.0,
    "sensor_width_mm": 36.0,
    #: The checklist's own worked example: "high, ~30 cm above eye level, angled down ~20".
    "rig_rise_m": 0.300,
    #: Two phones started by hand are never simultaneous. Giving the raised camera a known
    #: head start exercises the audio sync path instead of assuming it, and gives us a true
    #: offset to check the correlator against.
    "start_offset_s": 1.0,
    "samples": 32,
    "render": False,
}


def log(message):
    print("PGH " + str(message), flush=True)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


# -- materials ---------------------------------------------------------------------------
#
# Both materials exist to make something matchable. Default Suzanne is smooth and one flat
# grey, which is unreconstructable by definition: feature matching needs albedo variation,
# and no amount of geometry detail substitutes for it. Bump alone will not do either -- it
# only modulates shading, and under even lighting that is a weak signal.


def _grain(nodes, links, coords, scale, amount):
    """A signed high-frequency perturbation, ready to add to a colour.

    Built from Math nodes rather than a Mix node on purpose: ``ShaderNodeMix`` addresses its
    colour inputs by an index that moved between Blender versions, and getting it wrong
    fails silently as a flat surface rather than loudly as an error.
    """
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.65
    links.new(coords, noise.inputs["Vector"])

    centre = nodes.new("ShaderNodeMath")
    centre.operation = "SUBTRACT"
    centre.inputs[1].default_value = 0.5
    links.new(noise.outputs["Fac"], centre.inputs[0])

    gain = nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = amount
    links.new(centre.outputs[0], gain.inputs[0])
    return gain.outputs[0], noise.outputs["Fac"]


def _patchwork(name, cell_scale, gain, lift, grains, roughness, bump):
    """A Voronoi patchwork, tone-mapped into a usable range.

    ``Voronoi``'s colour output is uniform random over the full unit cube, which renders as
    blown-out confetti. ``gain`` and ``lift`` map it linearly into a band, which is what
    keeps highlights off the ceiling. Compressing the range costs contrast, so the grain is
    added *after* the map rather than before, where it would be squashed with everything else.

    Both take a per-channel triple, and that is the point rather than a convenience: giving
    the subject and the room the same treatment camouflaged one against the other, and a
    reconstruction that cannot tell them apart is not a hard test, it is a broken one. Warm
    and saturated for the subject against cool and drab for the room separates them by hue
    while leaving the room every bit as feature-dense as the solver needs it to be.
    """
    gain = (gain, gain, gain) if isinstance(gain, (int, float)) else tuple(gain)
    lift = (lift, lift, lift) if isinstance(lift, (int, float)) else tuple(lift)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    # Matte on purpose. A glossy surface shades differently from every viewpoint, which
    # breaks the brightness constancy dense matching assumes -- the same mechanism that made
    # cup-1's glass table reconstruct badly.
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.12
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tex = nodes.new("ShaderNodeTexCoord")
    # Generated coordinates are normalised over the bounding box, so a scale means the same
    # thing whatever size the object ends up being.
    coords = tex.outputs["Generated"]

    cells = nodes.new("ShaderNodeTexVoronoi")
    cells.voronoi_dimensions = "3D"
    cells.feature = "F1"
    cells.inputs["Scale"].default_value = cell_scale
    links.new(coords, cells.inputs["Vector"])

    scaled = nodes.new("ShaderNodeVectorMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = gain
    links.new(cells.outputs["Color"], scaled.inputs[0])

    lifted = nodes.new("ShaderNodeVectorMath")
    lifted.operation = "ADD"
    lifted.inputs[1].default_value = lift
    links.new(scaled.outputs["Vector"], lifted.inputs[0])

    # More than one grain scale, because a detector only sees structure at the scale it is
    # looking for. Flat patch interiors give SIFT nothing but the patch boundaries to work
    # with, and dense matching nothing to correlate at all: the first pass here had exactly
    # that failing, visible at 100% as large areas of uniform colour.
    surface = lifted.outputs["Vector"]
    finest = None
    for scale, amount in grains:
        grain, finest = _grain(nodes, links, coords, scale, amount)
        grained = nodes.new("ShaderNodeVectorMath")
        grained.operation = "ADD"
        links.new(surface, grained.inputs[0])
        links.new(grain, grained.inputs[1])
        surface = grained.outputs["Vector"]
    links.new(surface, bsdf.inputs["Base Color"])

    if bump > 0.0 and finest is not None:
        weave = nodes.new("ShaderNodeBump")
        weave.inputs["Strength"].default_value = bump
        links.new(finest, weave.inputs["Height"])
        links.new(weave.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def fabric_material():
    """Bold swatches of cloth, with a weave inside each.

    The two scales do different jobs. Large cells give each part of the surface an identity
    a matcher can recognise again after a half turn, which is what closes the loop; the fine
    weave gives dense matching something to correlate at pixel scale. Either alone is
    markedly worse than both.
    """
    return _patchwork(
        "SuzanneFabric",
        cell_scale=7.0,
        # Warm and saturated: reds, oranges and yellows carry most of the variance.
        gain=(0.98, 0.60, 0.38),
        lift=(0.09, 0.05, 0.04),
        # ~17 mm blobs for the detector to anchor on, then ~3 mm weave for dense matching.
        grains=((34.0, 0.30), (190.0, 0.55)),
        roughness=0.85,
        bump=0.25,
    )


def room_material(name, scale):
    """The adversary: a busy, feature-dense surface, static in the world.

    Deliberately harder than a real room, but not the *same* as the subject. An earlier pass
    gave both the identical treatment and Suzanne vanished into the backdrop -- which tests
    nothing except whether two clones can be told apart. A real room is cluttered and
    comparatively drab; the subject is the saturated thing in front of it. So this is pushed
    down and towards grey, leaving plenty of texture to lock onto and a clear separation for
    the masker to find.
    """
    return _patchwork(
        name,
        cell_scale=scale,
        # Cool and desaturated: a room is cluttered but drab, and the subject is the
        # saturated thing in front of it.
        gain=(0.26, 0.34, 0.48),
        lift=(0.06, 0.08, 0.12),
        grains=((scale * 3.4, 0.55), (scale * 9.5, 0.62)),
        roughness=0.92,
        bump=0.0,
    )


# -- scene -------------------------------------------------------------------------------


def evaluated_dimensions(obj):
    """Bounding-box extents of the object with its modifiers applied, in local units."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        xs = [v.co.x for v in mesh.vertices]
        ys = [v.co.y for v in mesh.vertices]
        zs = [v.co.z for v in mesh.vertices]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    finally:
        evaluated.to_mesh_clear()


def add_subject(cfg):
    """A head on shoulders, which is what the real capture is pointed at.

    Suzanne alone is 834 mm ear to ear against a 600 mm height -- a landscape subject in a
    portrait frame, filling about a quarter of it and leaving the rest to backdrop. Giving
    her a torso makes the silhouette taller than it is wide, so the framing fills the way a
    head-and-shoulders scan actually does, and it puts the fabric swatches where clothing
    would be. The ears stay, and stay thin: they are the analogue of the ears/hair/chin the
    face-scan plan worries about losing.
    """
    bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0.0, 0.0, 0.0))
    head = bpy.context.active_object
    head.name = "Head"

    bpy.ops.mesh.primitive_cone_add(
        vertices=32, radius1=1.15, radius2=0.52, depth=1.7, location=(0.0, 0.0, -1.45)
    )
    torso = bpy.context.active_object
    torso.name = "Torso"

    # Joined rather than parented, and overlapping at the neck, so the true surface is one
    # shell -- otherwise the export stage's "keep the largest component" would be choosing
    # between a head and a body.
    bpy.ops.object.select_all(action="DESELECT")
    torso.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = "Subject"
    # Put the origin at the middle of the bust, so the spin axis runs through its centre the
    # way a chair's does, and so placing the origin at eye height places the *subject* there.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    bpy.ops.object.shade_smooth()

    # Scale from the *evaluated* height so the subdivision shrinkage is included, and the
    # figure written into scene.json describes the surface that actually gets rendered
    # rather than the control cage.
    bpy.context.view_layer.update()
    height = evaluated_dimensions(obj)[2]
    obj.scale = (cfg["subject_height_m"] / height,) * 3
    obj.location = tuple(cfg["subject_centre_m"])
    bpy.context.view_layer.update()

    obj.data.materials.append(fabric_material())
    return obj


def swept_radius(obj, axis_xy):
    """Furthest any point on the subject gets from the rotation axis, in world units.

    Not half the bounding box: the silhouette is widest somewhere around 45 degrees, where
    width and depth combine. Twice this is the widest the subject will ever appear, and it
    is what the framing has to accommodate.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = obj.matrix_world
        best = 0.0
        for vertex in mesh.vertices:
            world = matrix @ vertex.co
            best = max(best, math.hypot(world.x - axis_xy[0], world.y - axis_xy[1]))
        return best
    finally:
        evaluated.to_mesh_clear()


def framing_distance(cfg, radius):
    """How far back the rig has to sit for the whole sweep to stay in frame.

    The short side of a portrait frame is the binding constraint, so the fill target is
    measured against width; the subject is comfortably inside the long side either way.
    """
    needed_width = 2.0 * radius / cfg["target_fill"]
    return needed_width * cfg["lens_mm"] / cfg["sensor_width_mm"]


def add_backdrop(cfg):
    """Floor and back wall, both nailed to the world while the subject turns."""
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(room_material("FloorNoise", 55.0))

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 1.8, 1.5))
    wall = bpy.context.active_object
    wall.name = "Backdrop"
    wall.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    wall.data.materials.append(room_material("BackdropNoise", 45.0))
    return [floor, wall]


def add_lights(cfg):
    """Fixed in the world, not parented to the subject -- and that is the point.

    Room lights do not turn with you. The shading on the subject therefore changes as it
    rotates, which is a real difficulty for dense matching and one the actual capture will
    have. Parenting the lights to the subject would quietly delete it from the experiment.
    """
    centre = mathutils.Vector(cfg["subject_centre_m"])
    placements = [
        ("KeyLight", (-1.4, -1.2, 1.9), 45.0, 2.0),
        ("FillLight", (1.5, -1.0, 1.5), 26.0, 2.2),
        ("RimLight", (0.4, 1.2, 2.2), 16.0, 1.6),
    ]
    made = []
    for name, loc, power, size in placements:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = power
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = loc
        direction = centre - mathutils.Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        made.append(obj)

    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Ambient")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.30, 0.33, 0.38, 1.0)
        background.inputs["Strength"].default_value = 0.45
    return made


def add_rig(cfg):
    """Two cameras on one mount, which is the only way this app can recover absolute scale.

    ``orient.baseline_scale`` pairs camera centres that share a timeline slot and divides the
    measured lens-to-lens distance by the median model-space distance between them. One
    camera means no pairs, which means no scale, which means the export ships in arbitrary
    units with a warning. Parenting both to a single empty makes the separation rigid by
    construction, exactly as two phones clamped to one bar would be.
    """
    centre = mathutils.Vector(cfg["subject_centre_m"])
    rig = bpy.data.objects.new("Rig", None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.empty_display_size = 0.2
    bpy.context.collection.objects.link(rig)
    rig.location = (0.0, -cfg["camera_distance_m"], centre.z)

    cameras = {}
    for name, rise in (("cam-eye", 0.0), ("cam-high", cfg["rig_rise_m"])):
        data = bpy.data.cameras.new(name)
        data.lens = cfg["lens_mm"]
        # Fit the sensor to the horizontal explicitly. Left on AUTO the fit follows whichever
        # dimension is larger, so a portrait frame silently changes what the focal length
        # means and the intrinsics we write down stop describing the ones we rendered with.
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = cfg["sensor_width_mm"]
        data.shift_x = 0.0
        data.shift_y = 0.0
        data.clip_start = 0.05
        data.clip_end = 100.0
        data.dof.use_dof = False

        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.parent = rig
        obj.location = (0.0, 0.0, rise)
        bpy.context.view_layer.update()
        # Aim at the subject centre. For the raised camera that lands about 22 degrees below
        # level, which is the checklist's "angled down ~20" without having to dial it in.
        direction = centre - obj.matrix_world.translation
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cameras[name] = obj

    bpy.context.view_layer.update()
    return rig, cameras


def action_fcurves(obj):
    """Every f-curve of an object's action, across both action APIs.

    Blender 4.4 moved actions to layers and slots, and ``Action.fcurves`` went away with the
    change. Reaching through the new structure keeps this working on 5.x without giving up
    on the 4.2 install that is also on this machine.
    """
    animation = obj.animation_data
    if animation is None or animation.action is None:
        return []
    action = animation.action
    if hasattr(action, "fcurves"):
        return list(action.fcurves)

    curves = []
    slot = getattr(animation, "action_slot", None)
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot) if slot is not None else None
            if bag is not None:
                curves.extend(bag.fcurves)
    return curves


def animate(subject, cfg):
    total = cfg["frames"]
    subject.rotation_euler = (0.0, 0.0, 0.0)
    subject.keyframe_insert("rotation_euler", frame=1)
    subject.rotation_euler = (0.0, 0.0, math.radians(cfg["degrees_per_frame"] * total))
    subject.keyframe_insert("rotation_euler", frame=total + 1)
    # Linear, because Blender's default easing bunches viewpoints at both ends of the spin
    # and would leave the middle of the turn under-sampled for no reason.
    curves = action_fcurves(subject)
    if not curves:
        raise RuntimeError("the subject's rotation keyframes produced no f-curves to linearise")
    for fcurve in curves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
        fcurve.update()


def configure_render(cfg):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = cfg["fps"]
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = cfg["frames"]
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    # Blender 5 splits output into media types, and the file-format enum is filtered to the
    # current one -- so asking for PNG while the scene is on VIDEO fails with a bare "enum
    # not found". Set the media type first and the formats come back.
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    # Standard rather than AgX: a filmic curve compresses exactly the contrast that feature
    # detection lives on, and we would end up measuring the tone map instead of the rig.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except TypeError:
        pass
    try:
        scene.eevee.taa_render_samples = cfg["samples"]
    except AttributeError:
        pass


# -- ground truth ------------------------------------------------------------------------
#
# Blender cameras look down -Z with +Y up; COLMAP's look down +Z with +Y down. Every pose we
# write is therefore recorded twice: once in Blender's convention, and once already
# converted, so the comparison script never has to rediscover the handedness and no reader
# has to trust that it did.

BLENDER_TO_COLMAP = mathutils.Matrix.Diagonal((1.0, -1.0, -1.0)).to_3x3()


def subject_frame(cfg, frame):
    """The subject's own frame at *frame*: where the reconstruction's world actually sits.

    Deliberately rotation and translation only, with no scale -- inverting the object's full
    matrix would undo the subject's scale as well and silently rescale every pose.
    """
    theta = math.radians(cfg["degrees_per_frame"] * (frame - 1))
    return mathutils.Matrix.Translation(
        mathutils.Vector(cfg["subject_centre_m"])
    ) @ mathutils.Matrix.Rotation(theta, 4, "Z")


def intrinsics(cfg):
    width, height = cfg["resolution"]
    # sensor_fit is HORIZONTAL, so the focal length is defined against sensor width whatever
    # the aspect ratio, and square pixels make fy equal to fx.
    fx = cfg["lens_mm"] * width / cfg["sensor_width_mm"]
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "params": [fx, fx, width / 2.0, height / 2.0],
        "fx": fx,
        "fy": fx,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "lens_mm": cfg["lens_mm"],
        "sensor_width_mm": cfg["sensor_width_mm"],
        "sensor_fit": "HORIZONTAL",
    }


def camera_track(cfg, camera):
    """Every frame's camera pose in the subject's frame, in both conventions.

    The camera never moves in the world, so its literal transform is a constant and says
    nothing. What COLMAP reconstructs is the *relative* motion, which in the subject's frame
    is an orbit -- and that is what this returns.
    """
    world = camera.matrix_world.copy()
    track = []
    for frame in range(1, cfg["frames"] + 1):
        in_subject = subject_frame(cfg, frame).inverted() @ world
        centre = in_subject.to_translation()
        rotation_c2w = in_subject.to_3x3().normalized() @ BLENDER_TO_COLMAP
        rotation_w2c = rotation_c2w.transposed()
        quaternion = rotation_w2c.to_quaternion()
        translation = rotation_w2c @ -centre
        track.append(
            {
                "frame": frame,
                "time_s": (frame - 1) / float(cfg["fps"]),
                "subject_yaw_deg": cfg["degrees_per_frame"] * (frame - 1),
                "center": [centre.x, centre.y, centre.z],
                # COLMAP's order, scalar first -- note scipy's from_quat wants (x, y, z, w).
                "qvec": [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
                "tvec": [translation.x, translation.y, translation.z],
            }
        )
    return track


def write_ground_truth(cfg, subject, cameras, out_dir):
    truth_dir = os.path.join(out_dir, "groundtruth")
    os.makedirs(truth_dir, exist_ok=True)

    eye = cameras["cam-eye"].matrix_world.translation
    high = cameras["cam-high"].matrix_world.translation
    baseline_mm = (high - eye).length * 1000.0

    relative = cameras["cam-eye"].matrix_world.inverted() @ cameras["cam-high"].matrix_world
    down_angle_deg = math.degrees(
        (cameras["cam-high"].matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))).angle(
            mathutils.Vector((0.0, 1.0, 0.0))
        )
    )

    rig = {
        "baseline_mm": baseline_mm,
        "baseline_measured_from": "camera matrix_world translations",
        "cam_eye_world": [list(row) for row in cameras["cam-eye"].matrix_world],
        "cam_high_world": [list(row) for row in cameras["cam-high"].matrix_world],
        "cam_high_relative_to_eye": [list(row) for row in relative],
        "cam_high_down_angle_deg": down_angle_deg,
        "start_offset_s": cfg["start_offset_s"],
        "offset_applied_to": "cam-high",
    }
    with open(os.path.join(truth_dir, "rig.json"), "w", encoding="utf-8") as handle:
        json.dump(rig, handle, indent=2)

    shared = intrinsics(cfg)
    payload = {
        "convention": {
            "qvec": "(w, x, y, z), world-to-camera, COLMAP order",
            "tvec": "world-to-camera translation; center = -R^T t",
            "center": "camera position in the subject's frame",
            "axes": "converted from Blender (-Z forward, +Y up) to COLMAP (+Z forward, -Y up)",
        },
        "cameras": {
            name: {
                "intrinsics": shared,
                "clip_start_s": cfg["start_offset_s"] if name == "cam-high" else 0.0,
                "track": camera_track(cfg, obj),
            }
            for name, obj in cameras.items()
        },
    }
    with open(os.path.join(truth_dir, "cameras.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    dimensions = evaluated_dimensions(subject)
    scale = subject.scale.z
    width_px, height_px = cfg["resolution"]
    frame_width = cfg["camera_distance_m"] * cfg["sensor_width_mm"] / cfg["lens_mm"] * 1000.0
    frame_height = frame_width * height_px / width_px
    scene_facts = {
        "subject_height_mm": dimensions[2] * scale * 1000.0,
        "subject_width_mm": dimensions[0] * scale * 1000.0,
        "subject_depth_mm": dimensions[1] * scale * 1000.0,
        "subject_centre_m": list(cfg["subject_centre_m"]),
        "degrees_per_frame": cfg["degrees_per_frame"],
        "frames": cfg["frames"],
        "fps": cfg["fps"],
        "revolutions": cfg["degrees_per_frame"] * cfg["frames"] / 360.0,
        "resolution": list(cfg["resolution"]),
        "camera_distance_m": cfg["camera_distance_m"],
        "swept_diameter_mm": cfg["swept_radius_m"] * 2000.0,
        "frame_width_mm_at_subject": frame_width,
        "frame_height_mm_at_subject": frame_height,
        "subject_fill_width": (cfg["swept_radius_m"] * 2000.0) / frame_width,
        "subject_fill_height": dimensions[2] * scale * 1000.0 / frame_height,
        "lens_mm": cfg["lens_mm"],
        "blender": bpy.app.version_string,
    }
    with open(os.path.join(truth_dir, "scene.json"), "w", encoding="utf-8") as handle:
        json.dump(scene_facts, handle, indent=2)

    # The true surface, at rotation zero, with subdivision applied and the object's own
    # transform baked in, so it lands in the same units and the same place as the poses.
    bpy.context.scene.frame_set(1)
    bpy.ops.object.select_all(action="DESELECT")
    subject.select_set(True)
    bpy.context.view_layer.objects.active = subject
    bpy.ops.wm.ply_export(
        filepath=os.path.join(truth_dir, "suzanne_gt.ply"),
        export_selected_objects=True,
        apply_modifiers=True,
        export_normals=True,
        export_uv=False,
        export_colors="NONE",
        ascii_format=False,
    )
    return rig, scene_facts


# -- render ------------------------------------------------------------------------------


def render_all(cfg, cameras, out_dir):
    scene = bpy.context.scene
    written = {}
    for name, obj in cameras.items():
        scene.camera = obj
        target = os.path.join(out_dir, "frames-" + name)
        os.makedirs(target, exist_ok=True)
        # A trailing separator makes Blender name frames by number alone: 0001.png upward.
        scene.render.filepath = target + os.sep
        log("rendering " + name + " -> " + target)
        bpy.ops.render.render(animation=True)
        written[name] = target
    return written


# -- entry point -------------------------------------------------------------------------


def build(cfg):
    cfg = dict(CONFIG, **(cfg or {}))
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    subject = add_subject(cfg)
    cfg["swept_radius_m"] = swept_radius(subject, cfg["subject_centre_m"][:2])
    if cfg.get("camera_distance_m") is None:
        cfg["camera_distance_m"] = framing_distance(cfg, cfg["swept_radius_m"])
    add_backdrop(cfg)
    add_lights(cfg)
    rig_empty, cameras = add_rig(cfg)
    animate(subject, cfg)
    configure_render(cfg)
    bpy.context.scene.camera = cameras["cam-eye"]

    rig, scene_facts = write_ground_truth(cfg, subject, cameras, out_dir)

    blend_path = os.path.join(out_dir, "suzanne-turntable.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    result = {
        "blend": blend_path,
        "out_dir": out_dir,
        "rig": {k: rig[k] for k in ("baseline_mm", "cam_high_down_angle_deg", "start_offset_s")},
        "scene": scene_facts,
        "cameras": sorted(cameras),
        "rendered": None,
    }
    if cfg["render"]:
        result["rendered"] = render_all(cfg, cameras, out_dir)
    return result


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    cfg = {}
    for arg in argv:
        if arg == "--render":
            cfg["render"] = True
        elif not arg.startswith("--"):
            with open(arg, "r", encoding="utf-8") as handle:
                cfg.update(json.load(handle))
    print(RESULT_PREFIX + json.dumps(build(cfg)), flush=True)


if __name__ == "__main__" and "--" in sys.argv:
    main()
