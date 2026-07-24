"use client";

import { decode } from "@msgpack/msgpack";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Region = "EC" | "DG" | "CA3" | "CA1";
type ViewMode = "functional" | "illustrative";
type GeometryPayload = {
  schema?: number;
  circuit_version: string;
  neuron_count: number;
  positions: number[][];
  layout?: {
    kind: string;
    authority: string;
    distance_semantics: boolean;
    default_view: string;
    notice: string;
  };
  regions: Record<Region, { start: number; end: number; count?: number; role?: string }>;
  pathways: {
    name: string;
    source: Region;
    target: Region;
    fanout: number;
    synapse_count: number;
    inhibitory: boolean;
    plastic: boolean;
    recurrent: boolean;
    rendering?: string;
  }[];
};
type Frame = {
  step: number;
  phase?: string;
  active_neurons: number[];
  active_edges: [number, number, number, string][];
  region_spikes: Record<Region, number>;
  memory_id?: number;
  engram_id?: string;
};
type Selection = {
  kind: "neuron" | "engram" | "none";
  id?: number | string;
  region?: Region;
};
type NeuronDetail = {
  neuron_id: number;
  region: Region;
  incoming_total: number;
  outgoing_total: number;
  incoming: { neuron_id: number; weight: number; pathway: string; inhibitory: boolean }[];
  outgoing: { neuron_id: number; weight: number; pathway: string; inhibitory: boolean }[];
  incoming_truncated: boolean;
  outgoing_truncated: boolean;
  time_cell: { preferred_phase: number; width: number } | null;
};

const API = "http://127.0.0.1:8765";
const REGION_COLOR: Record<Region, number> = {
  EC: 0x44d7b6,
  DG: 0xf4c95d,
  CA3: 0xff756d,
  CA1: 0x8ea7ff,
};
const REGION_LABEL: Record<Region, string> = {
  EC: "Entorhinal cortex",
  DG: "Dentate gyrus",
  CA3: "CA3 recurrent field",
  CA1: "CA1 output",
};
const REGION_ROLE: Record<Region, string> = {
  EC: "Context and cortical input",
  DG: "Sparse pattern separation",
  CA3: "Recurrent association and pattern completion",
  CA1: "Comparison and governed readout",
};
const FUNCTIONAL_CENTER: Record<Region, [number, number, number]> = {
  EC: [-4.8, 0, 0],
  DG: [-1.7, 0.35, 0.35],
  CA3: [1.6, 0.35, 0],
  CA1: [4.7, 0, -0.25],
};

function pathwayRegions(name: string): [Region, Region] {
  if (name.endsWith("_INHIBITION")) {
    const region = name.replace("_INHIBITION", "") as Region;
    return [region, region];
  }
  const [source, target] = name.split("_") as [Region, Region];
  return [source, target];
}

function functionalPosition(id: number, region: Region): [number, number, number] {
  let seed = (id + 1) * 2654435761;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const theta = random() * Math.PI * 2;
  const phi = Math.acos(2 * random() - 1);
  const radius = Math.cbrt(random());
  const center = FUNCTIONAL_CENTER[region];
  const scale = region === "DG" ? [1.25, 1.05, 0.75] : [1.05, 0.85, 0.65];
  return [
    center[0] + Math.sin(phi) * Math.cos(theta) * radius * scale[0],
    center[1] + Math.sin(phi) * Math.sin(theta) * radius * scale[1],
    center[2] + Math.cos(phi) * radius * scale[2],
  ];
}

function fallbackGeometry(): GeometryPayload {
  const sizes: Record<Region, number> = {
    EC: 8192,
    DG: 16384,
    CA3: 8192,
    CA1: 4096,
  };
  const centers: Record<Region, [number, number, number]> = {
    EC: [-3.4, 0, 0],
    DG: [-1.2, 0.3, 0.8],
    CA3: [1, 0.5, 0.2],
    CA1: [3.2, 0, -0.4],
  };
  const regions = {} as GeometryPayload["regions"];
  const positions: number[][] = [];
  let seed = 41;
  let cursor = 0;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  (Object.keys(sizes) as Region[]).forEach((region) => {
    const start = cursor;
    for (let index = 0; index < sizes[region]; index += 1) {
      const theta = random() * Math.PI * 2;
      const radius = 0.35 + random() * 0.9;
      const center = centers[region];
      positions.push([
        center[0] + Math.cos(theta) * radius,
        center[1] + Math.sin(theta) * radius,
        center[2] + (random() - 0.5) * 0.5,
      ]);
      cursor += 1;
    }
    regions[region] = { start, end: cursor };
  });
  return {
    circuit_version: "trisynaptic-v1 / waiting for service",
    neuron_count: positions.length,
    positions,
    regions,
    pathways: [
      ["EC_DG", 65536],
      ["DG_CA3", 98304],
      ["EC_CA3", 32768],
      ["CA3_CA3", 32768],
      ["CA3_CA1", 65536],
      ["EC_CA1", 32768],
    ].map(([name, synapse_count]) => {
      const [source, target] = pathwayRegions(String(name));
      return {
      name: String(name),
      source,
      target,
      fanout: Number(synapse_count) / sizes[source],
      synapse_count: Number(synapse_count),
      inhibitory: false,
      plastic: true,
      recurrent: source === target,
    }}),
    layout: {
      kind: "fallback-illustrative",
      authority: "visual-only",
      distance_semantics: false,
      default_view: "functional-topology",
      notice: "Fallback coordinates are visual scaffolding only.",
    },
  };
}

function regionFor(id: number, regions: GeometryPayload["regions"]): Region {
  return (
    (Object.entries(regions) as [Region, { start: number; end: number }][]).find(
      ([, range]) => id >= range.start && id < range.end,
    )?.[0] ?? "EC"
  );
}

function normalizeGeometry(payload: GeometryPayload): GeometryPayload {
  return {
    ...payload,
    pathways: payload.pathways.map((pathway) => {
      const [source, target] = pathwayRegions(pathway.name);
      return {
        ...pathway,
        source: pathway.source ?? source,
        target: pathway.target ?? target,
        fanout:
          pathway.fanout ??
          pathway.synapse_count /
            Math.max(1, payload.regions[source].end - payload.regions[source].start),
        recurrent: pathway.recurrent ?? source === target,
      };
    }),
  };
}

export default function NeuralObservatory() {
  const mount = useRef<HTMLDivElement>(null);
  const frameRef = useRef<Frame | null>(null);
  const geometryRef = useRef<GeometryPayload | null>(null);
  const selectRef = useRef<(selection: Selection) => void>(() => {});
  const [connection, setConnection] = useState<"connecting" | "live" | "offline">(
    "connecting",
  );
  const [rendererKind, setRendererKind] = useState("initializing");
  const [geometry, setGeometry] = useState<GeometryPayload | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [selection, setSelection] = useState<Selection>({ kind: "none" });
  const [playing, setPlaying] = useState(true);
  const [showSynapses, setShowSynapses] = useState(true);
  const [showInhibition, setShowInhibition] = useState(true);
  const [showPathways, setShowPathways] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("functional");
  const [neuronDetail, setNeuronDetail] = useState<NeuronDetail | null>(null);
  const [showGuide, setShowGuide] = useState(false);
  const [recording, setRecording] = useState("Live stream");
  const [recordings, setRecordings] = useState<{ name: string; bytes: number }[]>([]);
  const [recordedFrames, setRecordedFrames] = useState<Frame[]>([]);
  const [timeline, setTimeline] = useState(100);

  useEffect(() => {
    selectRef.current = setSelection;
  }, []);

  useEffect(() => {
    if (window.localStorage.getItem("cmf-observatory-guide") !== "seen") {
      queueMicrotask(() => setShowGuide(true));
    }
  }, []);

  useEffect(() => {
    if (selection.kind !== "neuron" || typeof selection.id !== "number") {
      return;
    }
    let cancelled = false;
    fetch(`${API}/neuron/${selection.id}`)
      .then((response) => {
        if (!response.ok) throw new Error("connectivity unavailable");
        return response.json();
      })
      .then((payload: NeuronDetail) => {
        if (!cancelled) setNeuronDetail(payload);
      })
      .catch(() => {
        if (!cancelled) setNeuronDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selection]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/geometry`)
      .then((response) => {
        if (!response.ok) throw new Error("geometry unavailable");
        return response.json();
      })
      .then((payload: GeometryPayload) => {
        if (!cancelled) setGeometry(normalizeGeometry(payload));
      })
      .catch(() => {
        if (!cancelled) setGeometry(fallbackGeometry());
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!geometry) return;
    geometryRef.current = geometry;
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;
    if (recording !== "Live stream") return;
    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket("ws://127.0.0.1:8765/live");
      socket.binaryType = "arraybuffer";
      socket.onopen = () => setConnection("live");
      socket.onmessage = (event) => {
        if (!playing) return;
        const next = decode(new Uint8Array(event.data)) as Frame;
        frameRef.current = next;
        setFrame(next);
        setTimeline(100);
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        setConnection("offline");
        retry = setTimeout(connect, 2500);
      };
    };
    connect();
    return () => {
      clearTimeout(retry);
      socket?.close();
    };
  }, [geometry, playing, recording]);

  useEffect(() => {
    fetch(`${API}/recordings`)
      .then((response) => response.json())
      .then((payload) => setRecordings(payload.recordings ?? []))
      .catch(() => setRecordings([]));
  }, [connection]);

  useEffect(() => {
    if (recording === "Live stream") return;
    let cancelled = false;
    fetch(`${API}/recordings/${encodeURIComponent(recording)}`)
      .then((response) => response.arrayBuffer())
      .then((buffer) => {
        const bytes = new Uint8Array(buffer);
        const magic = new TextDecoder().decode(bytes.slice(0, 7));
        if (magic !== "HMREC1\n") throw new Error("invalid recording");
        const view = new DataView(buffer);
        const frames: Frame[] = [];
        let offset = 7;
        while (offset + 4 <= bytes.length && frames.length < 20_000) {
          const length = view.getUint32(offset, true);
          offset += 4;
          if (length > 16 * 1024 * 1024 || offset + length > bytes.length) break;
          frames.push(decode(bytes.slice(offset, offset + length)) as Frame);
          offset += length;
        }
        if (!cancelled) {
          setRecordedFrames(frames);
          const last = frames[frames.length - 1] ?? null;
          frameRef.current = last;
          setFrame(last);
          setTimeline(100);
          setPlaying(false);
        }
      })
      .catch(() => {
        if (!cancelled) setRecordedFrames([]);
      });
    return () => {
      cancelled = true;
    };
  }, [recording]);

  useEffect(() => {
    if (!playing || !recordedFrames.length || recording === "Live stream") return;
    const timer = window.setInterval(() => {
      setTimeline((current) => {
        const currentIndex = Math.round(
          (current / 100) * (recordedFrames.length - 1),
        );
        const nextIndex = Math.min(recordedFrames.length - 1, currentIndex + 1);
        const nextFrame = recordedFrames[nextIndex];
        frameRef.current = nextFrame;
        setFrame(nextFrame);
        if (nextIndex === recordedFrames.length - 1) setPlaying(false);
        return (nextIndex / Math.max(1, recordedFrames.length - 1)) * 100;
      });
    }, 80);
    return () => window.clearInterval(timer);
  }, [playing, recordedFrames, recording]);

  useEffect(() => {
    if (!mount.current || !geometry) return;
    const container = mount.current;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | {
      domElement: HTMLCanvasElement;
      setSize: (w: number, h: number) => void;
      setPixelRatio: (ratio: number) => void;
      setAnimationLoop: (callback: (() => void) | null) => void;
      render: (scene: THREE.Scene, camera: THREE.Camera) => void;
      dispose: () => void;
      init?: () => Promise<void>;
    };
    let controls: OrbitControls;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05090d);
    scene.fog = new THREE.FogExp2(0x05090d, 0.028);
    const camera = new THREE.PerspectiveCamera(52, 1, 0.01, 120);
    camera.position.set(0, 3.8, 12);
    const keys = new Set<string>();

    const renderedPositions = geometry.positions.map((position, id) =>
      viewMode === "functional"
        ? functionalPosition(id, regionFor(id, geometry.regions))
        : (position as [number, number, number]),
    );
    const positions = new Float32Array(geometry.neuron_count * 3);
    const colors = new Float32Array(geometry.neuron_count * 3);
    renderedPositions.forEach((position, id) => {
      positions.set(position, id * 3);
      const color = new THREE.Color(REGION_COLOR[regionFor(id, geometry.regions)]);
      colors.set([color.r * 0.58, color.g * 0.58, color.b * 0.58], id * 3);
    });
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    pointGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const pointMaterial = new THREE.PointsMaterial({
      size: 0.032,
      vertexColors: true,
      transparent: true,
      opacity: 0.74,
      sizeAttenuation: true,
    });
    const points = new THREE.Points(pointGeometry, pointMaterial);
    scene.add(points);

    const clusterGroup = new THREE.Group();
    (Object.entries(geometry.regions) as [
      Region,
      { start: number; end: number },
    ][]).forEach(([region, range]) => {
      const center = new THREE.Vector3();
      for (let id = range.start; id < range.end; id += 128) {
        center.add(
          new THREE.Vector3(
            renderedPositions[id][0],
            renderedPositions[id][1],
            renderedPositions[id][2],
          ),
        );
      }
      center.divideScalar(Math.ceil((range.end - range.start) / 128));
      const mesh = new THREE.Mesh(
        new THREE.IcosahedronGeometry(
          0.42 + Math.cbrt(range.end - range.start) / 52,
          2,
        ),
        new THREE.MeshBasicMaterial({
          color: REGION_COLOR[region],
          wireframe: true,
          transparent: true,
          opacity: 0.6,
        }),
      );
      mesh.position.copy(center);
      mesh.userData.region = region;
      clusterGroup.add(mesh);
    });
    clusterGroup.visible = false;
    scene.add(clusterGroup);

    const pathwayGroup = new THREE.Group();
    geometry.pathways.forEach((pathway) => {
      if (pathway.inhibitory && !showInhibition) return;
      const source = new THREE.Vector3(...FUNCTIONAL_CENTER[pathway.source]);
      const target = new THREE.Vector3(...FUNCTIONAL_CENTER[pathway.target]);
      const color = pathway.inhibitory ? 0xff756d : REGION_COLOR[pathway.source];
      if (pathway.source === pathway.target) {
        if (pathway.inhibitory) {
          const field = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.03, 2),
            new THREE.MeshBasicMaterial({
              color,
              wireframe: true,
              transparent: true,
              opacity: 0.09,
            }),
          );
          field.position.copy(source);
          if (pathway.source === "DG") field.scale.set(1.2, 1.05, 0.78);
          else field.scale.set(1.02, 0.86, 0.68);
          field.userData.pathway = pathway.name;
          pathwayGroup.add(field);
          return;
        }
        const loop = new THREE.Mesh(
          new THREE.TorusGeometry(1.15, 0.018, 6, 64),
          new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: pathway.inhibitory ? 0.24 : 0.48,
          }),
        );
        loop.position.copy(source);
        loop.rotation.x = Math.PI / 2;
        loop.userData.pathway = pathway.name;
        pathwayGroup.add(loop);
        return;
      }
      const direction = target.clone().sub(source);
      const length = direction.length();
      direction.normalize();
      const arrow = new THREE.ArrowHelper(
        direction,
        source,
        length,
        color,
        0.28,
        0.15,
      );
      arrow.userData.pathway = pathway.name;
      arrow.line.material.transparent = true;
      arrow.line.material.opacity = pathway.inhibitory ? 0.3 : 0.52;
      pathwayGroup.add(arrow);
    });
    pathwayGroup.visible = showPathways && viewMode === "functional";
    scene.add(pathwayGroup);

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(0), 3),
    );
    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x8fe6d1,
      transparent: true,
      opacity: 0.24,
      blending: THREE.AdditiveBlending,
    });
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);
    scene.add(new THREE.AmbientLight(0xa9c8ff, 0.9));
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.09 };
    const pointer = new THREE.Vector2();

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
      renderer?.setSize(width, height);
    };
    const onPointer = (event: PointerEvent) => {
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(points, false)[0];
      if (hit?.index !== undefined) {
        selectRef.current({
          kind: "neuron",
          id: hit.index,
          region: regionFor(hit.index, geometry.regions),
        });
      }
    };
    const onKeyDown = (event: KeyboardEvent) => keys.add(event.key.toLowerCase());
    const onKeyUp = (event: KeyboardEvent) => keys.delete(event.key.toLowerCase());

    const initialize = async () => {
      const forceWebGL = new URLSearchParams(window.location.search).has("webgl");
      if ("gpu" in navigator && !forceWebGL) {
        try {
          const webgpuModule = await import("three/webgpu");
          const webgpu = new webgpuModule.WebGPURenderer({ antialias: true });
          await webgpu.init();
          renderer = webgpu;
          setRendererKind("WebGPU");
        } catch {
          renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
          setRendererKind("WebGL2 fallback");
        }
      } else {
        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        setRendererKind("WebGL2 fallback");
      }
      if (disposed) {
        renderer.dispose();
        return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      container.appendChild(renderer.domElement);
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.055;
      controls.screenSpacePanning = true;
      renderer.domElement.addEventListener("pointerdown", onPointer);
      window.addEventListener("resize", resize);
      window.addEventListener("keydown", onKeyDown);
      window.addEventListener("keyup", onKeyUp);
      resize();

      let previousActive: number[] = [];
      renderer.setAnimationLoop(() => {
        const current = frameRef.current;
        const colorAttribute = pointGeometry.getAttribute("color") as THREE.BufferAttribute;
        previousActive.forEach((id) => {
          const base = new THREE.Color(REGION_COLOR[regionFor(id, geometry.regions)]);
          colorAttribute.setXYZ(id, base.r * 0.58, base.g * 0.58, base.b * 0.58);
        });
        if (current) {
          current.active_neurons.forEach((id) => {
            colorAttribute.setXYZ(id, 1, 1, 0.9);
          });
          previousActive = current.active_neurons;
          colorAttribute.needsUpdate = true;
          if (showSynapses) {
            const visibleEdges = current.active_edges.filter(
              (edge) => showInhibition || !edge[3].includes("INHIBITION"),
            );
            const edgePositions = new Float32Array(
              Math.min(visibleEdges.length, 4000) * 6,
            );
            visibleEdges.slice(0, 4000).forEach((edge, index) => {
              const source = renderedPositions[edge[0]];
              const target = renderedPositions[edge[1]];
              if (source && target) {
                edgePositions.set([...source, ...target], index * 6);
              }
            });
            lineGeometry.setAttribute(
              "position",
              new THREE.BufferAttribute(edgePositions, 3),
            );
          }
        }
        lines.visible = showSynapses;
        pathwayGroup.visible = showPathways && viewMode === "functional";
        const distance = camera.position.distanceTo(controls.target);
        points.visible = distance < 19;
        clusterGroup.visible = distance >= 19;
        const speed = 0.055;
        if (keys.has("w")) camera.translateZ(-speed);
        if (keys.has("s")) camera.translateZ(speed);
        if (keys.has("a")) camera.translateX(-speed);
        if (keys.has("d")) camera.translateX(speed);
        if (keys.has("q")) camera.translateY(-speed);
        if (keys.has("e")) camera.translateY(speed);
        controls.update();
        renderer.render(scene, camera);
      });
    };
    initialize();
    return () => {
      disposed = true;
      renderer?.setAnimationLoop(null);
      renderer?.domElement.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      controls?.dispose();
      pointGeometry.dispose();
      pointMaterial.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      pathwayGroup.traverse((object) => {
        const item = object as THREE.Mesh;
        item.geometry?.dispose();
        if (Array.isArray(item.material)) item.material.forEach((value) => value.dispose());
        else item.material?.dispose();
      });
      renderer?.dispose();
      if (renderer?.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [geometry, showInhibition, showPathways, showSynapses, viewMode]);

  const selectRegion = useCallback(
    (region: Region) => setSelection({ kind: "neuron", region }),
    [],
  );
  const totalSynapses = useMemo(
    () => geometry?.pathways.reduce((sum, item) => sum + item.synapse_count, 0) ?? 0,
    [geometry],
  );

  return (
    <main className="observatory">
      <header className="topbar">
        <div>
          <p className="eyebrow">Hermes memory architecture</p>
          <h1>Neural Observatory</h1>
        </div>
        <div className="system-readout">
          <span className={`signal ${connection}`} />
          <div>
            <strong>{connection === "live" ? "Live circuit" : "Local preview"}</strong>
            <small>{rendererKind} · read-only</small>
          </div>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel left-panel">
          <section>
            <p className="section-label">Circuit</p>
            <div className="metric-pair">
              <div><strong>{geometry?.neuron_count.toLocaleString() ?? "—"}</strong><span>neurons</span></div>
              <div><strong>{totalSynapses.toLocaleString()}</strong><span>synapses</span></div>
            </div>
          </section>
          <section>
            <p className="section-label">Regions</p>
            <div className="region-list">
              {(Object.keys(REGION_COLOR) as Region[]).map((region) => (
                <button key={region} onClick={() => selectRegion(region)}>
                  <span
                    className="region-mark"
                    style={{ backgroundColor: `#${REGION_COLOR[region].toString(16)}` }}
                  />
                  <span><strong>{region}</strong><small>{REGION_LABEL[region]}</small></span>
                  <em>{frame?.region_spikes?.[region] ?? 0}</em>
                </button>
              ))}
            </div>
          </section>
          <section>
            <p className="section-label">View</p>
            <div className="mode-switch" role="group" aria-label="Circuit layout">
              <button
                className={viewMode === "functional" ? "active" : ""}
                onClick={() => setViewMode("functional")}
              >
                Functional
              </button>
              <button
                className={viewMode === "illustrative" ? "active" : ""}
                onClick={() => setViewMode("illustrative")}
              >
                Illustrative
              </button>
            </div>
            <p className="layout-explainer">
              {viewMode === "functional"
                ? "Regions are separated for legibility. Arrows show implemented aggregate pathways."
                : "Deterministic simulation coordinates. Distance and ring shape have no anatomical meaning."}
            </p>
            <label className="switch-row">
              <input
                type="checkbox"
                checked={showPathways}
                onChange={(event) => setShowPathways(event.target.checked)}
              />
              Aggregate pathways
            </label>
            <label className="switch-row">
              <input
                type="checkbox"
                checked={showSynapses}
                onChange={(event) => setShowSynapses(event.target.checked)}
              />
              Measured active edges
            </label>
            <label className="switch-row">
              <input
                type="checkbox"
                checked={showInhibition}
                onChange={(event) => setShowInhibition(event.target.checked)}
              />
              Inhibitory field
            </label>
          </section>
          <section>
            <p className="section-label">Pathway key</p>
            <div className="truth-key">
              <p><i className="key-line aggregate" />Aggregate configured pathway</p>
              <p><i className="key-line measured" />Exact edge active in this frame</p>
              <p><i className="key-line inhibitory" />Local inhibitory/recurrent field</p>
            </div>
          </section>
          <section className="controls-help">
            <p className="section-label">Navigation</p>
            <p>Drag to orbit · right-drag to pan · scroll to zoom</p>
            <p>W A S D + Q E for free flight</p>
          </section>
        </aside>

        <div className="viewport">
          <div ref={mount} className="canvas-mount" />
          <div className={`truth-banner ${viewMode}`}>
            <strong>
              {viewMode === "functional" ? "Functional topology" : "Illustrative coordinates"}
            </strong>
            <span>
              Neuron positions are visual scaffolding · arrows are configured pathways ·
              flashes and thin edges are measured telemetry
            </span>
            <button onClick={() => setShowGuide(true)}>How to read this</button>
          </div>
          <div className="viewport-caption">
            <span>
              {geometry?.circuit_version}
              {(geometry?.schema ?? 1) < 2 ? " · legacy geometry schema" : " · schema 2"}
            </span>
            <span>Step {frame?.step?.toLocaleString() ?? "—"}</span>
          </div>
          {connection !== "live" && (
            <div className="offline-notice">
              <strong>Telemetry service is offline</strong>
              <span>Fallback geometry only; no measured firing or active edges.</span>
            </div>
          )}
          {showGuide && (
            <div className="guide-card" role="dialog" aria-label="How to read the neural observatory">
              <p className="section-label">What is real here?</p>
              <h2>Topology is authoritative. Geometry is not.</h2>
              <ol>
                <li><strong>Volumes</strong> group actual neuron IDs by implemented region.</li>
                <li><strong>Large arrows</strong> summarize configured source→target pathways and synapse counts.</li>
                <li><strong>Bright points and thin edges</strong> are exact activity measured in the selected frame.</li>
                <li><strong>Illustrative mode</strong> exposes the deterministic coordinates used only to place points.</li>
              </ol>
              <button onClick={() => {
                window.localStorage.setItem("cmf-observatory-guide", "seen");
                setShowGuide(false);
              }}>Understood</button>
            </div>
          )}
        </div>

        <aside className="panel inspector">
          <p className="section-label">Inspector</p>
          {selection.kind === "none" ? (
            <div className="empty-inspector">
              <div className="crosshair">+</div>
              <h2>Select a neuron</h2>
              <p>Click a point to inspect its region, firing state, and connected trace.</p>
            </div>
          ) : (
            <div className="selection-card">
              <p className="selection-type">{selection.kind}</p>
              <h2>{selection.id !== undefined ? `#${selection.id}` : selection.region}</h2>
              <dl>
                <div><dt>Region</dt><dd>{selection.region ?? "—"}</dd></div>
                <div><dt>Function</dt><dd>{selection.region ? REGION_ROLE[selection.region] : "—"}</dd></div>
                <div><dt>State</dt><dd>{typeof selection.id === "number" && frame?.active_neurons.includes(selection.id) ? "Firing" : "Quiescent"}</dd></div>
                <div><dt>Phase</dt><dd>{frame?.phase ?? "Wake / live"}</dd></div>
                <div><dt>Plasticity</dt><dd>Local STDP</dd></div>
                {neuronDetail?.neuron_id === selection.id && (
                  <>
                    <div><dt>Incoming</dt><dd>{neuronDetail.incoming_total.toLocaleString()} exact edges</dd></div>
                    <div><dt>Outgoing</dt><dd>{neuronDetail.outgoing_total.toLocaleString()} exact edges</dd></div>
                    <div><dt>Time cell</dt><dd>{neuronDetail.time_cell ? `phase ${neuronDetail.time_cell.preferred_phase}` : "No"}</dd></div>
                  </>
                )}
              </dl>
              {neuronDetail?.neuron_id === selection.id && (
                <div className="edge-sample">
                  <p className="section-label">Bounded adjacency sample</p>
                  {[...neuronDetail.outgoing.slice(0, 3), ...neuronDetail.incoming.slice(0, 3)].map((edge, index) => (
                    <p key={`${edge.pathway}-${edge.neuron_id}-${index}`}>
                      <span>{edge.pathway}</span>
                      <strong>#{edge.neuron_id}</strong>
                      <em>{edge.weight.toFixed(3)}</em>
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}
          <section className="flow">
            <p className="section-label">Connection matrix</p>
            <div className="pathway-matrix">
              {geometry?.pathways.map((pathway) => (
                <div key={pathway.name} className={pathway.inhibitory ? "inhibitory" : ""}>
                  <span>{pathway.source} → {pathway.target}</span>
                  <strong>{pathway.synapse_count.toLocaleString()}</strong>
                  <em>{pathway.inhibitory ? "inhibitory" : pathway.recurrent ? "recurrent" : `fanout ${pathway.fanout}`}</em>
                </div>
              ))}
            </div>
          </section>
          <section className="provenance">
            <p className="section-label">Provenance</p>
            <p>No raw transcripts are sent to this viewer. Memory evidence is requested only when an engram is selected.</p>
          </section>
        </aside>
      </section>

      <footer className="timeline">
        <button className="transport" onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play"}>
          {playing ? "Ⅱ" : "▶"}
        </button>
        <select
          value={recording}
          onChange={(event) => {
            setRecording(event.target.value);
            setRecordedFrames([]);
          }}
        >
          <option>Live stream</option>
          {recordings.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name.replace(/^sleep-/, "Sleep ").replace(/\.hmrec$/, "")}
            </option>
          ))}
        </select>
        <div className="scrubber">
          <input
            aria-label="Replay timeline"
            type="range"
            min="0"
            max="100"
            value={timeline}
            onChange={(event) => {
              const value = Number(event.target.value);
              setTimeline(value);
              setPlaying(false);
              if (recordedFrames.length) {
                const index = Math.round(
                  (value / 100) * (recordedFrames.length - 1),
                );
                const selectedFrame = recordedFrames[index];
                frameRef.current = selectedFrame;
                setFrame(selectedFrame);
              }
            }}
          />
          <div><span>T−60s</span><span>{frame?.phase?.toUpperCase() ?? "LIVE"}</span><span>Now</span></div>
        </div>
        <div className="activity-count">
          <strong>{frame?.active_neurons.length.toLocaleString() ?? 0}</strong>
          <span>active now</span>
        </div>
      </footer>
    </main>
  );
}
