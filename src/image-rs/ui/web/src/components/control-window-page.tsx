import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MenuShell } from "@/components/menu-shell";
import { tauriInvoke, tauriListen, type OpenResult } from "@/lib/tauri";
import type { CommandDispatchEvent, TopLevelMenu } from "@/types/menu";

type ControlWindowPageProps = {
  menus: TopLevelMenu[];
  implementedCommands: Set<string>;
  onDispatch: (event: CommandDispatchEvent) => void;
};

export function ControlWindowPage({ menus, implementedCommands, onDispatch }: ControlWindowPageProps) {
  const [path, setPath] = useState("");
  const [result, setResult] = useState<OpenResult | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | null = null;

    tauriListen<OpenResult>("launcher-open-result", (event) => {
      setResult(event.payload);
    }).then((cleanup) => {
      unlisten = cleanup;
    });

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  const summary = useMemo(() => {
    if (!result) {
      return "Waiting for image input.";
    }
    return `Opened ${result.opened.length}, focused ${result.focused.length}, skipped ${result.skipped.length}, errors ${result.errors.length}`;
  }, [result]);

  async function openFromPath() {
    const normalized = path.trim();
    if (!normalized) {
      return;
    }
    const payload = await tauriInvoke<OpenResult>("open_images", { paths: [normalized] });
    setResult(payload);
  }

  async function executeImplementedCommand(commandId: string) {
    switch (commandId) {
      case "file.open":
        document.getElementById("launcher-path-input")?.focus();
        break;
      case "window.next":
        await tauriInvoke<string>("cycle_window", { direction: 1 });
        break;
      case "window.previous":
        await tauriInvoke<string>("cycle_window", { direction: -1 });
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

  const listItems = [
    ...(result?.opened ?? []).map((item) => `Opened: ${item}`),
    ...(result?.focused ?? []).map((item) => `Focused: ${item}`),
    ...(result?.skipped ?? []).map((item) => `Skipped: ${item}`),
    ...(result?.errors ?? []).map((item) => `Error: ${item}`),
  ];

  return (
    <main className="flex min-h-screen flex-col bg-muted/30">
      <MenuShell
        menus={menus}
        onDispatch={(event) => void handleDispatch(event)}
        implementedCommands={implementedCommands}
      />

      <section className="mx-auto mt-8 w-full max-w-xl rounded-lg border bg-card p-5 shadow-sm">
        <h1 className="text-xl font-semibold">ImageJ-rs</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Drop TIFF files on this window, or open by path.
        </p>

        <div className="mt-4 rounded-md border border-dashed bg-muted/40 p-6 text-center text-sm">
          <p className="font-medium">Drop TIFF Images Here</p>
          <p className="mt-1 text-muted-foreground">Duplicate files focus existing viewer windows.</p>
        </div>

        <div className="mt-4 flex gap-2">
          <Input
            id="launcher-path-input"
            value={path}
            onChange={(event) => setPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                void openFromPath();
              }
            }}
            placeholder="C:\\path\\to\\image.tif"
          />
          <Button onClick={() => void openFromPath()}>Open</Button>
        </div>

        <div className="mt-4 rounded-md border bg-background">
          <p className="border-b px-3 py-2 text-xs text-muted-foreground">{summary}</p>
          <ul className="max-h-48 list-disc overflow-auto px-6 py-3 text-xs">
            {listItems.length === 0 ? <li className="text-muted-foreground">No events yet.</li> : null}
            {listItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
