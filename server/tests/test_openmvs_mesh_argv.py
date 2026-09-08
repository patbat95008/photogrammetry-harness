"""The three mesh tools accept less, and different things, than their help suggests.

Everything asserted here was read off the v2.4.0 binaries in ``tools/`` and then
confirmed by running them against the cup-1 dense scene. The ``openMVS - Source/``
checkout in this repo tracks ``develop`` and is months ahead of them, so it is not
evidence about what these builders may emit.

Three traps, all of which fail silently rather than loudly:

* ``ReconstructMesh`` and ``RefineMesh`` write PLY for any export type they do not
  recognise, so asking either of them for ``glb`` yields a file the caller is not
  looking for and a stage that reports the tool produced nothing;
* the dense stage's ``.mvs`` carries only the sparse cloud, so a reconstruction without
  ``-p`` quietly uses a fifteenth of the points it should;
* ``RefineMesh`` alone defaults to the CPU, and a builder that "helpfully" defaulted it
  to the GPU would change what the engine does without anybody asking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pgh.vendor import openmvs


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve tool paths without needing the binaries present."""
    monkeypatch.setattr(openmvs, "_tool", lambda key: Path(f"C:/tools/{key}.exe"))


def value_after(argv: list, flag: str):
    """The argument following ``flag``, which is how every OpenMVS option is passed."""
    return argv[argv.index(flag) + 1]


# -- what may be exported ----------------------------------------------------


@pytest.mark.parametrize("builder", ["reconstruct_mesh", "refine_mesh"])
def test_reconstruct_and_refine_refuse_glb(builder: str) -> None:
    """Neither would refuse it themselves; they would write PLY under a .glb name."""
    kwargs = {"input_file": Path("s.mvs"), "output_file": Path("o.ply")}
    if builder == "refine_mesh":
        kwargs["mesh_file"] = Path("m.ply")

    with pytest.raises(ValueError, match="ply or obj"):
        getattr(openmvs, builder)(export_type="glb", **kwargs)


def test_only_texture_can_export_glb() -> None:
    argv = openmvs.texture_mesh(
        input_file=Path("s.mvs"),
        mesh_file=Path("m.ply"),
        output_file=Path("t.glb"),
        export_type="glb",
    )
    assert value_after(argv, "--export-type") == "glb"


def test_every_builder_accepts_ply_and_obj() -> None:
    for export_type in ("ply", "obj"):
        openmvs.reconstruct_mesh(
            input_file=Path("s.mvs"), output_file=Path("o.ply"), export_type=export_type
        )
        openmvs.refine_mesh(
            input_file=Path("s.mvs"),
            mesh_file=Path("m.ply"),
            output_file=Path("o.ply"),
            export_type=export_type,
        )


# -- the dense cloud has to be passed explicitly -----------------------------


def test_reconstruct_passes_the_dense_cloud_explicitly() -> None:
    """The .mvs holds only the sparse cloud; the dense points are a separate file.

    ``scene_dense.mvs`` reports "50189 points, 0 vertices, 0 faces" when it loads, while
    the 738,015 dense points sit in ``scene_dense.ply``. Reconstructing without ``-p``
    succeeds and produces a mesh -- from the sparse cloud, with nothing in the log
    calling it a mistake.
    """
    argv = openmvs.reconstruct_mesh(
        input_file=Path("scene_dense.mvs"),
        output_file=Path("mesh.ply"),
        point_cloud_file=Path("scene_dense.ply"),
    )
    assert value_after(argv, "-p") == Path("scene_dense.ply")


def test_the_cloud_flag_is_omitted_when_there_is_no_cloud() -> None:
    argv = openmvs.reconstruct_mesh(
        input_file=Path("scene.mvs"), output_file=Path("mesh.ply")
    )
    assert "-p" not in argv


# -- the CPU/GPU asymmetry ---------------------------------------------------


def test_refine_defaults_to_the_processor() -> None:
    """-2 is the binary's own default, alone among the three. Mirror it, don't improve it."""
    argv = openmvs.refine_mesh(
        input_file=Path("s.mvs"), mesh_file=Path("m.ply"), output_file=Path("r.ply")
    )
    assert value_after(argv, "--cuda-device") == "-2"


def test_the_other_two_default_to_the_best_gpu() -> None:
    reconstruct = openmvs.reconstruct_mesh(
        input_file=Path("s.mvs"), output_file=Path("o.ply")
    )
    texture = openmvs.texture_mesh(
        input_file=Path("s.mvs"), mesh_file=Path("m.ply"), output_file=Path("t.ply")
    )
    assert value_after(reconstruct, "--cuda-device") == "-1"
    assert value_after(texture, "--cuda-device") == "-1"


def test_refining_on_the_gpu_is_asked_for_explicitly() -> None:
    argv = openmvs.refine_mesh(
        input_file=Path("s.mvs"),
        mesh_file=Path("m.ply"),
        output_file=Path("r.ply"),
        cuda_device=-1,
    )
    assert value_after(argv, "--cuda-device") == "-1"


# -- size knobs --------------------------------------------------------------


def test_both_size_knobs_are_emitted() -> None:
    """The engine folds them into one target, with target-face-num winning when set.

    Both are emitted at their defaults so the tool's own "Command line:" log line is a
    complete record of the configuration a run used.
    """
    argv = openmvs.reconstruct_mesh(
        input_file=Path("s.mvs"),
        output_file=Path("o.ply"),
        target_face_num=250_000,
        decimate=0.5,
    )
    assert value_after(argv, "--target-face-num") == "250000"
    assert value_after(argv, "--decimate") == "0.5"


# -- the shape every builder shares ------------------------------------------


def all_builders() -> list[list]:
    return [
        openmvs.reconstruct_mesh(input_file=Path("s.mvs"), output_file=Path("o.ply")),
        openmvs.refine_mesh(
            input_file=Path("s.mvs"), mesh_file=Path("m.ply"), output_file=Path("r.ply")
        ),
        openmvs.texture_mesh(
            input_file=Path("s.mvs"), mesh_file=Path("m.ply"), output_file=Path("t.ply")
        ),
    ]


def test_the_binary_leads_the_argv() -> None:
    for argv in all_builders():
        assert isinstance(argv[0], Path)
        assert argv[0].suffix == ".exe"


def test_paths_are_passed_unstringified() -> None:
    """proc stringifies at the boundary; a builder that did it early loses Path checks."""
    for argv in all_builders():
        assert value_after(argv, "-i") == Path("s.mvs")


def test_the_working_folder_is_optional() -> None:
    for argv in all_builders():
        assert "-w" not in argv

    argv = openmvs.texture_mesh(
        input_file=Path("s.mvs"),
        mesh_file=Path("m.ply"),
        output_file=Path("t.ply"),
        working_folder=Path("D:/scratch"),
    )
    assert value_after(argv, "-w") == Path("D:/scratch")
    assert argv[-2] == "-w", "the working folder is appended last, as densify does"


def test_booleans_become_one_and_zero() -> None:
    argv = openmvs.reconstruct_mesh(
        input_file=Path("s.mvs"),
        output_file=Path("o.ply"),
        free_space_support=True,
        remove_spikes=False,
        crop_to_roi=False,
    )
    assert value_after(argv, "--free-space-support") == "1"
    assert value_after(argv, "--remove-spikes") == "0"
    assert value_after(argv, "--crop-to-roi") == "0"


def test_verbosity_is_never_emitted() -> None:
    """-v changes what the tools write to disk, not just what they say about it."""
    for argv in all_builders():
        assert "-v" not in argv and "--verbosity" not in argv


# -- masks (HANDOVER 6.7, and the collision -m would cause) -------------------


def test_densify_reads_masks_only_when_given_a_label() -> None:
    """--ignore-mask-label defaults to -1, which means "estimate one" and ignores
    the files on disk entirely. Passing -m without this reads no mask at all."""
    argv = openmvs.densify_point_cloud(
        input_file=Path("scene.mvs"), output_file=Path("dense.ply"),
        ignore_mask_label=0,
    )

    assert "--ignore-mask-label" in argv
    assert argv[argv.index("--ignore-mask-label") + 1] == "0"


def test_densify_says_nothing_about_masks_on_an_unmasked_run() -> None:
    argv = openmvs.densify_point_cloud(
        input_file=Path("scene.mvs"), output_file=Path("dense.ply")
    )

    assert "--ignore-mask-label" not in argv


def test_densify_refuses_the_mask_path_flag() -> None:
    """-m flattens every camera group into one directory keyed on the bare filename.

    The shared slot clock then guarantees cam_high/000042.jpg and cam_eye/000042.jpg
    both resolve to 000042.mask.png, so one camera's masks are applied to the other.
    It looks perfect on a single-camera run and is wrong on every rig, which is
    exactly the kind of omission somebody would helpfully "fix" later.
    """
    with pytest.raises(ValueError) as caught:
        openmvs.densify_point_cloud(
            input_file=Path("scene.mvs"), output_file=Path("dense.ply"),
            mask_path=Path("masks"),
        )

    message = str(caught.value)
    assert "one flat folder" in message
    assert "ignore_mask_label" in message, "the error must name the alternative"
