/**
 * The mesh, in three shadings, because one of them lies.
 *
 * A photographic texture is extremely convincing and hides exactly the faults worth
 * finding: a surface that is lumpy, doubled, or invented where a hole was closed still
 * looks like a photograph of the real thing. Matte shading with a moving light is what
 * exposes it, and the wireframe says how much of the detail is real geometry rather
 * than picture. That is why the mode switch is the first control here rather than a
 * decoration -- see HANDOVER 8 and the original design's UI table.
 *
 * Textured mode is deliberately UNLIT. The texture already contains the lighting from
 * the shoot, so lighting it again doubles the shadows and, worse, softens the seams
 * texturing produces. OpenMVS marks its glTF material KHR_materials_unlit for the same
 * reason.
 *
 * Framing, canvas sizing and the flip come from viewer3d.tsx and must agree with the
 * point-cloud viewer: the flip is the orientation the export stage will apply.
 */
import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import {
  CAMERA,
  Extent,
  FitToExtent,
  FlippedGroup,
  GL,
  robustExtent,
  useFlip,
  useSizedRef,
} from "./viewer3d";

export type Shading = "textured" | "matte" | "wireframe";

/**
 * Above this, wait to be asked. The bytes arrive over loopback in well under a second;
 * what costs is decoding and the video memory the texture takes once uploaded, and an
 * 8192-square atlas is a quarter of a gigabyte of it.
 */
const ASK_FIRST_BYTES = 40_000_000;

function extentOf(object: THREE.Object3D): Extent {
  const merged: number[] = [];
  object.updateWorldMatrix(true, true);
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    const position = mesh.geometry?.getAttribute?.("position");
    if (!position) return;
    const vertex = new THREE.Vector3();
    // Stride: the extent only needs enough samples for stable percentiles, and a
    // half-million-vertex mesh does not need every one of them copied out.
    const stride = Math.max(1, Math.floor(position.count / 40000));
    for (let i = 0; i < position.count; i += stride) {
      vertex.fromBufferAttribute(position as THREE.BufferAttribute, i);
      vertex.applyMatrix4(mesh.matrixWorld);
      merged.push(vertex.x, vertex.y, vertex.z);
    }
  });
  return robustExtent(new Float32Array(merged));
}

/**
 * Give the geometry normals if it arrived without any.
 *
 * OpenMVS writes its glTF with POSITION and TEXCOORD_0 and nothing else -- it marks the
 * material KHR_materials_unlit, so as far as the file is concerned normals would never
 * be read. They are exactly what the matte and wireframe modes need, though, and a lit
 * material with no normals to shade does not fail: it renders a flat black silhouette,
 * which reads as a catastrophically bad reconstruction rather than as a missing
 * attribute. Computing them here costs one pass and only happens when they are absent
 * (the export stage's mesh comes back from Blender already carrying them).
 */
function ensureNormals(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry && !mesh.geometry.hasAttribute("normal")) {
      mesh.geometry.computeVertexNormals();
    }
  });
}

/** Free the GPU memory a loaded mesh holds. An unfreed atlas is ~268 MB of VRAM. */
function disposeObject(object: THREE.Object3D | null) {
  object?.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose();
    const materials = ([] as THREE.Material[]).concat(mesh.material ?? []);
    for (const material of materials) {
      const map = (material as THREE.MeshStandardMaterial).map;
      map?.dispose();
      material.dispose();
    }
  });
}

/**
 * Swap every material for the one this shading mode calls for.
 *
 * The loaded materials are kept so textured mode can go back to them; the two derived
 * modes build their own and dispose them when the mode changes.
 */
function Shaded({
  object,
  shading,
}: {
  object: THREE.Object3D;
  shading: Shading;
}) {
  const original = useMemo(() => {
    const map = new Map<string, THREE.Material | THREE.Material[]>();
    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh) map.set(mesh.uuid, mesh.material);
    });
    return map;
  }, [object]);

  useEffect(() => {
    const derived: THREE.Material[] = [];

    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;

      if (shading === "textured") {
        const restored = original.get(mesh.uuid);
        if (restored) mesh.material = restored;
        return;
      }

      const material = new THREE.MeshStandardMaterial({
        color: 0xb8b8b8,
        roughness: 1,
        metalness: 0,
        wireframe: shading === "wireframe",
        side: THREE.DoubleSide,
      });
      derived.push(material);
      mesh.material = material;
    });

    return () => {
      for (const material of derived) material.dispose();
    };
  }, [object, shading, original]);

  return <primitive object={object} />;
}

export default function MeshViewer({
  meshUrl,
  format,
  runId,
  sizeBytes,
  faces,
  allowFlip = true,
  height = 560,
}: {
  meshUrl: string;
  format: "glb" | "gltf" | "obj" | "ply";
  runId?: string;
  /**
   * Whether turning the model over is still an open question. It is for the mesh, which
   * is in COLMAP's arbitrary frame; it is not for an export, which has already had the
   * orientation applied to it. Offering it there would turn an upright model upside down
   * and, because flip_x is in the export stage's fingerprint, mark the export stale for
   * having looked at it.
   */
  allowFlip?: boolean;
  /** Mesh plus texture. Above the threshold the viewer asks before loading. */
  sizeBytes?: number;
  faces?: number;
  height?: number;
}) {
  const [object, setObject] = useState<THREE.Object3D | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Matte, not textured. A photogrammetric albedo is often dark, so the textured
  // view can open near-black and read as a failed reconstruction; the page's own
  // copy tells you to judge the surface matte first. Textured is one click away.
  const [shading, setShading] = useState<Shading>("matte");
  const [dark, setDark] = useState(true);
  const [storedFlip, toggleFlip] = useFlip(allowFlip ? runId : undefined);
  const flipped = allowFlip && storedFlip;
  const [wrapRef, ready] = useSizedRef();

  const heavy = (sizeBytes ?? 0) > ASK_FIRST_BYTES;
  const [asked, setAsked] = useState(false);
  const wanted = !heavy || asked;

  useEffect(() => {
    if (!wanted) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    const fail = () => {
      if (cancelled) return;
      setError("could not load the mesh");
      setLoading(false);
    };
    const done = (loaded: THREE.Object3D) => {
      if (cancelled) {
        disposeObject(loaded);
        return;
      }
      ensureNormals(loaded);
      setObject(loaded);
      setLoading(false);
    };

    if (format === "glb" || format === "gltf") {
      // No DRACOLoader: it needs a decoder served alongside the app, and OpenMVS does
      // not compress its geometry.
      new GLTFLoader().load(meshUrl, (gltf) => done(gltf.scene), undefined, fail);
    } else if (format === "obj") {
      // The .mtl names its texture by a relative path, so the loader needs the
      // directory the artifacts are served from to resolve it.
      const base = meshUrl.slice(0, meshUrl.lastIndexOf("/") + 1);
      new MTLLoader()
        .setResourcePath(base)
        .load(
          meshUrl.replace(/\.obj$/, ".mtl"),
          (materials) => {
            materials.preload();
            new OBJLoader()
              .setMaterials(materials)
              .load(meshUrl, done, undefined, fail);
          },
          undefined,
          // An OBJ without its material file is still worth showing untextured.
          () => new OBJLoader().load(meshUrl, done, undefined, fail),
        );
    } else {
      new PLYLoader().load(
        meshUrl,
        (geometry) => done(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial())),
        undefined,
        fail,
      );
    }

    return () => {
      cancelled = true;
    };
  }, [meshUrl, format, wanted]);

  useEffect(() => () => disposeObject(object), [object]);

  const extent = useMemo(() => (object ? extentOf(object) : null), [object]);
  const target = useMemo(
    () => extent?.centre ?? ([0, 0, 0] as [number, number, number]),
    [extent],
  );

  const untextured = format === "ply";
  const effective: Shading = untextured && shading === "textured" ? "matte" : shading;

  return (
    <div className="cloud-viewer">
      <div className="cloud-toolbar">
        <span className="cloud-stat">
          {faces ? `${faces.toLocaleString()} triangles` : "mesh"}
          {sizeBytes ? <> · {(sizeBytes / 1e6).toFixed(0)} MB</> : null}
        </span>
        <label className="cloud-control">
          Shading
          <select
            value={effective}
            onChange={(e) => setShading(e.target.value as Shading)}
            title="Check it matte before you believe the texture."
          >
            <option value="textured" disabled={untextured}>
              Textured
            </option>
            <option value="matte">Matte</option>
            <option value="wireframe">Wireframe</option>
          </select>
        </label>
        {allowFlip && (
          <label
            className="cloud-control"
            title="Turn the model over. Saved with the run, and used when it is exported."
          >
            <input
              type="checkbox"
              checked={flipped}
              onChange={(e) => toggleFlip(e.target.checked)}
            />
            Flip
          </label>
        )}
        <label className="cloud-control">
          <input type="checkbox" checked={dark} onChange={(e) => setDark(e.target.checked)} />
          Dark
        </label>
      </div>

      {untextured && (
        <p className="stage-note">
          This mesh was written as PLY, which carries no texture, so only the untextured
          shadings are available. Re-run with the GLB or OBJ export type to see it
          painted.
        </p>
      )}

      <div className="cloud-canvas" style={{ height }} ref={wrapRef}>
        {!wanted && (
          <div className="cloud-status">
            <p>
              The mesh and its texture come to {((sizeBytes ?? 0) / 1e6).toFixed(0)} MB.
            </p>
            <button type="button" onClick={() => setAsked(true)}>
              Load it
            </button>
          </div>
        )}
        {wanted && loading && <div className="cloud-status">loading…</div>}
        {error && <div className="cloud-status cloud-error">{error}</div>}
        {object && ready && (
          <Canvas camera={CAMERA} gl={GL}>
            <color attach="background" args={[dark ? "#12161c" : "#eef1f5"]} />
            <FitToExtent extent={extent} />
            {/*
              Lights only matter to the two derived modes -- textured is unlit by
              design. The directional light is fixed in the scene rather than on the
              camera, so orbiting changes the angle it rakes across the surface at:
              a lumpy or doubled patch that hides at one angle catches the light at
              another, which is the whole point of looking at it matte.
            */}
            {effective !== "textured" && (
              <>
                <hemisphereLight args={[0xffffff, 0x40484f, 1.1]} />
                <directionalLight position={[2, 3, 4]} intensity={1.6} />
              </>
            )}
            <FlippedGroup target={target} flipped={flipped}>
              <Shaded object={object} shading={effective} />
            </FlippedGroup>
            <OrbitControls
              makeDefault
              enableDamping
              dampingFactor={0.12}
              target={target}
            />
          </Canvas>
        )}
      </div>
    </div>
  );
}
