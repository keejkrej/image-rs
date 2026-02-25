/* eslint-disable react-hooks/exhaustive-deps */
import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { MenuShell } from "@/components/menu-shell";
import { currentWindow, tauriInvoke, type ViewerFrameBuffer, type ViewerInit } from "@/lib/tauri";
import type { CommandDispatchEvent, TopLevelMenu } from "@/types/menu";

type ViewerWindowPageProps = {
  menus: TopLevelMenu[];
  implementedCommands: Set<string>;
  onDispatch: (event: CommandDispatchEvent) => void;
};

const INITIAL_PAN = 20;

export function ViewerWindowPage({ menus, implementedCommands, onDispatch }: ViewerWindowPageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const histogramRef = useRef<HTMLCanvasElement | null>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pathInputRef = useRef<HTMLInputElement | null>(null);

  const [viewerInit, setViewerInit] = useState<ViewerInit | null>(null);
  const [frame, setFrame] = useState<ViewerFrameBuffer | null>(null);
  const [z, setZ] = useState(0);
  const [t, setT] = useState(0);
  const [channel, setChannel] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(INITIAL_PAN);
  const [panY, setPanY] = useState(INITIAL_PAN);
  const [dragging, setDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState<{ x: number; y: number } | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; value: number } | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [openPath, setOpenPath] = useState("");
  const [status, setStatus] = useState("Ready.");
  const requestTokenRef = useRef(0);
  const skipNextFrameRequestRef = useRef(false);

  useEffect(() => {
    let mounted = true;

    tauriInvoke<ViewerInit>("viewer_init")
      .then((payload) => {
        if (!mounted) {
          return;
        }
        setViewerInit(payload);
        setFrame(payload.default_frame);
        skipNextFrameRequestRef.current = true;
        setOpenPath(payload.path);
      })
      .catch((error) => {
        setStatus(String(error));
      });

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!viewerInit) {
      return;
    }
    if (skipNextFrameRequestRef.current) {
      skipNextFrameRequestRef.current = false;
      return;
    }
    void requestFrame(z, t, channel);
  }, [viewerInit, z, t, channel]);

  useEffect(() => {
    if (!frame) {
      return;
    }

    const imageCanvas = document.createElement("canvas");
    imageCanvas.width = frame.width;
    imageCanvas.height = frame.height;
    const context = imageCanvas.getContext("2d");
    if (!context) {
      return;
    }

    const imageData = context.createImageData(frame.width, frame.height);
    for (let index = 0; index < frame.pixels_u8.length; index += 1) {
      const gray = frame.pixels_u8[index] ?? 0;
      const offset = index * 4;
      imageData.data[offset] = gray;
      imageData.data[offset + 1] = gray;
      imageData.data[offset + 2] = gray;
      imageData.data[offset + 3] = 255;
    }
    context.putImageData(imageData, 0, 0);
    imageCanvasRef.current = imageCanvas;

    drawHistogram(frame.histogram);
    redraw();
  }, [frame]);

  useEffect(() => {
    function onResize() {
      fitCanvas();
    }

    window.addEventListener("resize", onResize);
    fitCanvas();
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    redraw();
  }, [zoom, panX, panY]);

  function fitCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      redraw();
    }
  }

  async function requestFrame(nextZ: number, nextT: number, nextChannel: number) {
    const token = ++requestTokenRef.current;
    const startedAt = performance.now();
    try {
      const payload = await tauriInvoke<ViewerFrameBuffer>("viewer_frame_buffer", {
        request: { z: nextZ, t: nextT, channel: nextChannel },
      });

      if (token !== requestTokenRef.current) {
        return;
      }

      setFrame(payload);
      const elapsed = Math.round(performance.now() - startedAt);
      setStatus(`Rendered in ${elapsed} ms`);
    } catch (error) {
      setStatus(String(error));
    }
  }

  function drawHistogram(histogram: number[]) {
    const canvas = histogramRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const max = Math.max(...histogram, 1);
    const barWidth = canvas.width / histogram.length;
    context.fillStyle = "#475569";
    histogram.forEach((value, index) => {
      const ratio = value / max;
      const height = ratio * (canvas.height - 4);
      context.fillRect(index * barWidth, canvas.height - height, barWidth, height);
    });
  }

  function redraw() {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#0b0f1a";
    context.fillRect(0, 0, canvas.width, canvas.height);

    if (!imageCanvasRef.current) {
      return;
    }

    context.imageSmoothingEnabled = false;
    context.setTransform(zoom, 0, 0, zoom, panX, panY);
    context.drawImage(imageCanvasRef.current, 0, 0);
    context.setTransform(1, 0, 0, 1, 0, 0);
  }

  function updateHover(clientX: number, clientY: number) {
    if (!frame || !canvasRef.current) {
      setHover(null);
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    const imageX = Math.floor((localX - panX) / zoom);
    const imageY = Math.floor((localY - panY) / zoom);

    if (imageX < 0 || imageY < 0 || imageX >= frame.width || imageY >= frame.height) {
      setHover(null);
      return;
    }

    const index = imageY * frame.width + imageX;
    const gray = frame.pixels_u8[index] ?? 0;
    const value = frame.min + (gray / 255) * (frame.max - frame.min);
    setHover({ x: imageX, y: imageY, value });
  }

  async function executeImplementedCommand(commandId: string) {
    switch (commandId) {
      case "file.open":
        setOpenDialog(true);
        pathInputRef.current?.focus();
        break;
      case "file.close":
        await currentWindow().close();
        break;
      case "window.next":
        await tauriInvoke<string>("cycle_window", { direction: 1 });
        break;
      case "window.previous":
        await tauriInvoke<string>("cycle_window", { direction: -1 });
        break;
      case "image.zoom.in":
        setZoom((value) => Math.min(64, value * 1.2));
        break;
      case "image.zoom.out":
        setZoom((value) => Math.max(0.1, value / 1.2));
        break;
      case "image.zoom.reset":
        setZoom(1);
        setPanX(INITIAL_PAN);
        setPanY(INITIAL_PAN);
        break;
      default:
        break;
    }
  }

  async function handleDispatch(event: CommandDispatchEvent) {
    if (event.implemented) {
      await executeImplementedCommand(event.commandId);
    }
    onDispatch(event);
  }

  async function openPathFromDialog() {
    const trimmed = openPath.trim();
    if (!trimmed) {
      return;
    }
    try {
      await tauriInvoke("open_images", { paths: [trimmed] });
      setOpenDialog(false);
    } catch (error) {
      setStatus(String(error));
    }
  }

  const statusText = useMemo(() => {
    const hoverText = hover
      ? `X:${hover.x} Y:${hover.y} Value:${hover.value.toFixed(4)}`
      : "X:- Y:- Value:-";
    return `${hoverText}  Z:${z} T:${t} C:${channel}  Zoom:${(zoom * 100).toFixed(0)}%  ${status}`;
  }, [hover, z, t, channel, zoom, status]);

  const summary = viewerInit?.summary;

  return (
    <main className="flex min-h-screen flex-col bg-background">
      <MenuShell menus={menus} implementedCommands={implementedCommands} onDispatch={(event) => void handleDispatch(event)} />

      <section
        className={`grid min-h-0 flex-1 ${
          showInspector ? "grid-cols-[1fr_320px]" : "grid-cols-1"
        }`}
      >
        <div className="relative min-h-0 border-r bg-slate-950">
          <div className="absolute right-3 top-3 z-10">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setShowInspector((value) => !value)}
            >
              {showInspector ? "Hide Panel" : "Show Panel"}
            </Button>
          </div>
          <canvas
            ref={canvasRef}
            className="h-full w-full"
            onWheel={(event) => {
              event.preventDefault();
              const rect = event.currentTarget.getBoundingClientRect();
              const localX = event.clientX - rect.left;
              const localY = event.clientY - rect.top;
              const previousZoom = zoom;
              const nextZoom = Math.max(0.1, Math.min(64, event.deltaY < 0 ? previousZoom * 1.12 : previousZoom / 1.12));
              setPanX(localX - ((localX - panX) * nextZoom) / previousZoom);
              setPanY(localY - ((localY - panY) * nextZoom) / previousZoom);
              setZoom(nextZoom);
            }}
            onMouseDown={(event) => {
              setDragging(true);
              setLastMouse({ x: event.clientX, y: event.clientY });
            }}
            onMouseMove={(event) => {
              if (dragging && lastMouse) {
                const dx = event.clientX - lastMouse.x;
                const dy = event.clientY - lastMouse.y;
                setPanX((value) => value + dx);
                setPanY((value) => value + dy);
                setLastMouse({ x: event.clientX, y: event.clientY });
              }
              updateHover(event.clientX, event.clientY);
            }}
            onMouseUp={() => {
              setDragging(false);
              setLastMouse(null);
            }}
            onMouseLeave={() => {
              setDragging(false);
              setLastMouse(null);
              setHover(null);
            }}
          />
        </div>

        {showInspector ? (
          <aside className="space-y-3 overflow-auto bg-muted/20 p-3">
            <div className="rounded-md border bg-card p-3">
              <h1 className="text-sm font-semibold">
                {viewerInit?.path.split(/[\\/]/).pop() ?? "Viewer"}
              </h1>
              <p className="mt-1 break-all text-xs text-muted-foreground">
                {viewerInit?.path ?? "Loading..."}
              </p>
            </div>

            <div className="rounded-md border bg-card p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Slice
              </p>
              <div className="space-y-3">
                <label className="grid grid-cols-[18px_1fr_40px] items-center gap-2 text-xs">
                  <span>Z</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, (summary?.z_slices ?? 1) - 1)}
                    value={z}
                    onChange={(event) => setZ(Number(event.target.value))}
                  />
                  <span>{z}</span>
                </label>
                <label className="grid grid-cols-[18px_1fr_40px] items-center gap-2 text-xs">
                  <span>T</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, (summary?.times ?? 1) - 1)}
                    value={t}
                    onChange={(event) => setT(Number(event.target.value))}
                  />
                  <span>{t}</span>
                </label>
                <label className="grid grid-cols-[18px_1fr_40px] items-center gap-2 text-xs">
                  <span>C</span>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, (summary?.channels ?? 1) - 1)}
                    value={channel}
                    onChange={(event) => setChannel(Number(event.target.value))}
                  />
                  <span>{channel}</span>
                </label>
              </div>
            </div>

            <div className="rounded-md border bg-card p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Histogram
              </p>
              <canvas
                ref={histogramRef}
                width={280}
                height={140}
                className="w-full rounded border bg-white"
              />
            </div>

            <div className="rounded-md border bg-card p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Metadata
              </p>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-[11px]">
                {summary ? JSON.stringify(summary, null, 2) : "Loading..."}
              </pre>
            </div>
          </aside>
        ) : null}
      </section>

      <footer className="border-t bg-muted/20 px-3 py-2 text-xs text-muted-foreground">{statusText}</footer>

      <Dialog open={openDialog} onOpenChange={setOpenDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Open TIFF</DialogTitle>
            <DialogDescription>Enter a TIFF file path to open a new viewer window.</DialogDescription>
          </DialogHeader>
          <Input
            ref={pathInputRef}
            value={openPath}
            onChange={(event) => setOpenPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void openPathFromDialog();
              }
            }}
            placeholder="C:\\path\\to\\image.tif"
          />
          <DialogFooter>
            <Button variant="secondary" onClick={() => setOpenDialog(false)}>
              Cancel
            </Button>
            <Button onClick={() => void openPathFromDialog()}>Open</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}
