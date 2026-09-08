/**
 * The parts the point-cloud and mesh viewers must agree on.
 *
 * Extracted rather than copied, because each of these is a lesson that is silently
 * wrong when re-derived: the percentile framing, the two hoisted constants, the
 * canvas-sizing gate, the flip, and the rotation about the model's own centre. The
 * flip in particular is not style -- capture.flip_x is one run-level fact that the
 * export stage reads to orient what it writes, so two implementations that drift make
 * the exported orientation ambiguous.
 *
 * What is specific to points or to surfaces stays in the viewer that owns it.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useThree } from "@react-three/fiber";

export interface Extent {
  centre: [number, number, number];
  radius: number;
}

// Hoisted rather than written inline in the JSX. A fresh object literal on every
// render makes r3f tear down and rebuild the renderer, and the rebuilt canvas comes
// back at the HTML default of 300x150 and never resizes -- a viewer stuck in the
// corner of its own panel.
export const CAMERA = { fov: 55, position: [2, 2, 2] as [number, number, number] };
//: preserveDrawingBuffer keeps the composited frame readable, so the canvas can be
//: captured or saved. Without it a screenshot of the viewer comes out black.
export const GL = { preserveDrawingBuffer: true };

/**
 * Where the geometry actually is, ignoring the outliers every reconstruction produces.
 *
 * Fitting to the raw bounding box does not work here. A COLMAP sparse cloud always
 * carries a handful of badly-triangulated points flung far from the scene -- the cup
 * orbit spans 11 units of real content inside a 54-unit bounding box -- so a camera
 * framed on min/max leaves the subject a few percent of the view and looks, wrongly,
 * like an empty reconstruction. A mesh has the same problem for a different reason:
 * the Delaunay hull throws stray tetrahedra well outside the subject.
 *
 * Percentiles rather than a mean and standard deviation, because the outliers are
 * distant enough to drag both.
 */
export function robustExtent(positions: Float32Array): Extent {
  const count = Math.floor(positions.length / 3);
  if (count === 0) return { centre: [0, 0, 0], radius: 1 };

  // Sampling keeps this cheap on a million-point cloud; percentiles are stable.
  const stride = Math.max(1, Math.floor(count / 40000));
  const axes: number[][] = [[], [], []];
  for (let i = 0; i < count; i += stride) {
    axes[0].push(positions[i * 3]);
    axes[1].push(positions[i * 3 + 1]);
    axes[2].push(positions[i * 3 + 2]);
  }

  const centre: number[] = [];
  let span = 0;
  for (const values of axes) {
    values.sort((a, b) => a - b);
    const low = values[Math.floor(values.length * 0.02)];
    const high = values[Math.floor(values.length * 0.98)];
    centre.push((low + high) / 2);
    span = Math.max(span, high - low);
  }

  return { centre: centre as [number, number, number], radius: span / 2 || 1 };
}

/** Frame the camera on the geometry once, at a distance that fits its real extent. */
export function FitToExtent({ extent }: { extent: Extent | null }) {
  const { camera } = useThree();
  const fitted = useRef(false);

  useEffect(() => {
    if (!extent || fitted.current) return;
    const [cx, cy, cz] = extent.centre;
    const r = extent.radius;
    camera.position.set(cx + r * 2.0, cy + r * 1.3, cz + r * 2.0);
    camera.lookAt(cx, cy, cz);
    camera.near = r / 100;
    camera.far = r * 200;
    camera.updateProjectionMatrix();
    fitted.current = true;
  }, [extent, camera]);

  return null;
}

/**
 * Hold a canvas back until its container has a real size.
 *
 * The renderer measures its container on mount and starts its render loop from that,
 * so mounting into a container measured as zero leaves the canvas at the HTML default
 * of 300x150 with the loop never started. Measuring first and mounting second costs
 * one render and removes that possibility.
 *
 * Confirmed rendering in Firefox. It does not render in the embedded browser used by
 * some tooling, where ResizeObserver never fires at all and only a window resize event
 * ever produces a measurement -- so if a viewer ever looks blank, check it in a real
 * browser before assuming the reconstruction is empty.
 */
export function useSizedRef(): [React.RefObject<HTMLDivElement>, boolean] {
  const ref = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const wrap = ref.current;
    if (!wrap) return;
    const check = () => {
      const { width, height } = wrap.getBoundingClientRect();
      if (width > 0 && height > 0) setReady(true);
    };
    check();
    const observer = new ResizeObserver(check);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  return [ref, ready];
}

/**
 * Whether to stand the model up, read from and written back to the run.
 *
 * COLMAP's world frame is arbitrary and every capture so far has come out inverted, so
 * this starts on -- and starts on before the run has answered, so the first paint is
 * already the right way up rather than flipping under the viewer a moment later. The
 * toggle exists because the sign is genuinely ambiguous: the axis is recoverable from
 * the plane the cameras lie in, its direction is not.
 *
 * The write is optimistic because the toggle has to feel immediate, but reverted if it
 * fails: this setting is read back on the next visit and by the export stage, so a
 * checkbox that quietly did not persist would be a lie about the exported model.
 */
export function useFlip(runId?: string): [boolean, (next: boolean) => void] {
  const [flipped, setFlipped] = useState(true);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    fetch(`/api/runs/${runId}/orientation`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: { flip_x: boolean }) => {
        if (!cancelled) setFlipped(data.flip_x);
      })
      .catch(() => {
        /* the default is the answer for every run so far; show it and carry on */
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  function toggle(next: boolean) {
    setFlipped(next);
    if (!runId) return;
    fetch(`/api/runs/${runId}/orientation`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flip_x: next }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
      })
      .catch(() => setFlipped(!next));
  }

  return [flipped, toggle];
}

/**
 * The flip, as a rotation about the model's centre rather than the world origin.
 *
 * Turning the model about its own centre leaves that centre a fixed point, so the
 * framing and the OrbitControls target stay correct and toggling turns the model in
 * place instead of swinging it out of frame. The group stays mounted in both states,
 * so toggling costs a matrix update rather than a remount of the geometry.
 */
export function FlippedGroup({
  target,
  flipped,
  children,
}: {
  target: [number, number, number];
  flipped: boolean;
  children: React.ReactNode;
}) {
  // Memoised for the same reason as CAMERA and GL: a fresh array per render restarts
  // the scene graph node it is applied to.
  const rotation = useMemo<[number, number, number]>(
    () => (flipped ? [Math.PI, 0, 0] : [0, 0, 0]),
    [flipped],
  );
  const negTarget = useMemo<[number, number, number]>(
    () => [-target[0], -target[1], -target[2]],
    [target],
  );

  return (
    <group position={target} rotation={rotation}>
      <group position={negTarget}>{children}</group>
    </group>
  );
}
