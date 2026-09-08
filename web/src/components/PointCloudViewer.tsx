/**
 * One viewer, used by both the sparse and dense stages.
 *
 * Both stages write the same preview PLY layout -- xyz float32 plus rgb uchar,
 * decimated to a cap -- so a single loader path serves both and there is nothing to
 * guess about property order. The full-resolution cloud stays on disk.
 *
 * The camera frusta are the point of this component. Registration rate and
 * reprojection error tell you a model is wrong but never how: a background-locked
 * solve, an orbit that broke in half, a handful of views flung across the scene all
 * produce plausible numbers and unmistakable shapes. Drawing where the camera
 * thought it was, in the scene it reconstructed, is the cheapest way to see it.
 *
 * Framing, canvas sizing, the flip and the rotation about the model's centre are
 * shared with the mesh viewer and live in viewer3d.tsx.
 */
import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import {
  CAMERA,
  FitToExtent,
  FlippedGroup,
  GL,
  robustExtent,
  useFlip,
  useSizedRef,
} from "./viewer3d";

export interface PoseImage {
  name: string;
  center: [number, number, number];
  qvec: [number, number, number, number];
  num_points: number;
}

export interface PosesFile {
  images: PoseImage[];
  cameras: { camera_id: number; width: number; height: number; params: number[] }[];
}

function Cloud({
  geometry,
  pointSize,
}: {
  geometry: THREE.BufferGeometry;
  pointSize: number;
}) {
  // sizeAttenuation off: points keep a constant screen size, so a sparse region
  // reads as sparse instead of dissolving as you pull back.
  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: geometry.hasAttribute("color"),
        color: geometry.hasAttribute("color") ? 0xffffff : 0x9aa4b2,
        sizeAttenuation: false,
      }),
    [geometry, pointSize],
  );
  useEffect(() => () => material.dispose(), [material]);
  return <points geometry={geometry} material={material} />;
}

/**
 * Camera frusta, drawn as a wireframe pyramid per registered view.
 *
 * Scaled to a fraction of the scene rather than to anything metric: photogrammetry
 * recovers shape but not size, so there is no real-world length to draw them at.
 */
function Frusta({
  poses,
  scale,
  highlightWeak,
}: {
  poses: PoseImage[];
  scale: number;
  highlightWeak: boolean;
}) {
  const { geometry, colors } = useMemo(() => {
    const positions: number[] = [];
    const colorList: number[] = [];
    const weakThreshold = 40;

    for (const pose of poses) {
      const [w, x, y, z] = pose.qvec;
      // COLMAP stores camera-from-world; the drawing needs world-from-camera.
      const q = new THREE.Quaternion(x, y, z, w).invert();
      const origin = new THREE.Vector3(...pose.center);
      const weak = highlightWeak && pose.num_points < weakThreshold;
      const colour = weak ? new THREE.Color(0xe0794a) : new THREE.Color(0x5b9bd5);

      // Four corners of the image plane, one unit down the +Z (viewing) axis.
      const corners = [
        new THREE.Vector3(-0.5, -0.35, 1),
        new THREE.Vector3(0.5, -0.35, 1),
        new THREE.Vector3(0.5, 0.35, 1),
        new THREE.Vector3(-0.5, 0.35, 1),
      ].map((c) => c.multiplyScalar(scale).applyQuaternion(q).add(origin));

      const segments: [THREE.Vector3, THREE.Vector3][] = [
        [origin, corners[0]],
        [origin, corners[1]],
        [origin, corners[2]],
        [origin, corners[3]],
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
      ];
      for (const [a, b] of segments) {
        positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
        colorList.push(colour.r, colour.g, colour.b, colour.r, colour.g, colour.b);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colorList, 3));
    return { geometry: geo, colors: colorList };
  }, [poses, scale, highlightWeak]);

  useEffect(() => () => geometry.dispose(), [geometry]);
  if (!colors.length) return null;
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial vertexColors />
    </lineSegments>
  );
}

/**
 * The trajectory, in capture order: a closed orbit should read as a closed ring.
 *
 * Built as a plain THREE.Line and mounted through <primitive> rather than as a
 * <line> element -- in JSX that tag resolves to the SVG line, not the three.js one.
 */
function Trajectory({ poses }: { poses: PoseImage[] }) {
  const line = useMemo(() => {
    const ordered = [...poses].sort((a, b) => a.name.localeCompare(b.name));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        ordered.flatMap((p) => p.center),
        3,
      ),
    );
    return new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x6f7c8f }),
    );
  }, [poses]);

  useEffect(
    () => () => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
    },
    [line],
  );

  if (poses.length < 2) return null;
  return <primitive object={line} />;
}

export default function PointCloudViewer({
  cloudUrl,
  posesUrl,
  runId,
  height = 460,
}: {
  cloudUrl: string;
  posesUrl?: string;
  /** Enables the flip toggle to persist against the run. Without it, it is view-only. */
  runId?: string;
  height?: number;
}) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [poses, setPoses] = useState<PoseImage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pointSize, setPointSize] = useState(1.6);
  const [showFrusta, setShowFrusta] = useState(true);
  const [dark, setDark] = useState(true);
  const [flipped, toggleFlip] = useFlip(runId);
  const [wrapRef, ready] = useSizedRef();

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    new PLYLoader().load(
      cloudUrl,
      (geo) => {
        if (cancelled) return;
        geo.computeBoundingBox();
        setGeometry(geo);
        setLoading(false);
      },
      undefined,
      () => {
        if (cancelled) return;
        setError("could not load the point cloud");
        setLoading(false);
      },
    );

    return () => {
      cancelled = true;
    };
  }, [cloudUrl]);

  useEffect(() => {
    if (!posesUrl) return;
    let cancelled = false;
    fetch(posesUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: PosesFile) => {
        if (!cancelled) setPoses(data.images ?? []);
      })
      .catch(() => {
        /* frusta are an overlay; the cloud is still worth showing without them */
      });
    return () => {
      cancelled = true;
    };
  }, [posesUrl]);

  useEffect(() => () => geometry?.dispose(), [geometry]);

  const extent = useMemo(() => {
    const attr = geometry?.getAttribute("position");
    if (!attr) return null;
    return robustExtent(attr.array as Float32Array);
  }, [geometry]);
  const sceneRadius = extent?.radius ?? 1;
  // Memoised for the same reason as GL above: a new array on every render would restart
  // OrbitControls continuously.
  const target = useMemo(
    () => extent?.centre ?? ([0, 0, 0] as [number, number, number]),
    [extent],
  );

  const pointCount = geometry?.getAttribute("position")?.count ?? 0;
  const weak = poses.filter((p) => p.num_points < 40).length;

  return (
    <div className="cloud-viewer">
      <div className="cloud-toolbar">
        <span className="cloud-stat">
          {pointCount.toLocaleString()} points
          {poses.length > 0 && <> · {poses.length} cameras</>}
          {weak > 0 && <> · <span className="cloud-warn">{weak} weakly tied</span></>}
        </span>
        <label className="cloud-control">
          Point size
          <input
            type="range"
            min={0.5}
            max={6}
            step={0.1}
            value={pointSize}
            onChange={(e) => setPointSize(Number(e.target.value))}
          />
        </label>
        {poses.length > 0 && (
          <label className="cloud-control">
            <input
              type="checkbox"
              checked={showFrusta}
              onChange={(e) => setShowFrusta(e.target.checked)}
            />
            Cameras
          </label>
        )}
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
        <label className="cloud-control">
          <input type="checkbox" checked={dark} onChange={(e) => setDark(e.target.checked)} />
          Dark
        </label>
      </div>

      <div className="cloud-canvas" style={{ height }} ref={wrapRef}>
        {loading && <div className="cloud-status">loading…</div>}
        {error && <div className="cloud-status cloud-error">{error}</div>}
        {geometry && ready && (
          <Canvas camera={CAMERA} gl={GL}>
            <color attach="background" args={[dark ? "#12161c" : "#eef1f5"]} />
            <FitToExtent extent={extent} />
            {/*
              Cameras turn with the cloud: they are a claim about where each shot was
              taken from in this model, and are only true of it in the same frame.
            */}
            <FlippedGroup target={target} flipped={flipped}>
              <Cloud geometry={geometry} pointSize={pointSize} />
              {showFrusta && poses.length > 0 && (
                <>
                  <Frusta poses={poses} scale={sceneRadius * 0.08} highlightWeak />
                  <Trajectory poses={poses} />
                </>
              )}
            </FlippedGroup>
            {/*
              The target must be the cloud's centre. OrbitControls drives the camera
              every frame from its own target, which defaults to the origin -- so
              without this it silently undoes the framing above and points the camera
              at wherever the world happens to be zero.
            */}
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
