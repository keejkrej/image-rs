import menuManifest from "@/menu/menu-manifest.json";
import { ControlWindowPage } from "@/components/control-window-page";
import { ViewerWindowPage } from "@/components/viewer-window-page";
import { Toaster } from "@/components/ui/toaster";
import { ToastProvider, useToast } from "@/hooks/use-toast";
import type { CommandDispatchEvent, TopLevelMenu } from "@/types/menu";

const viewerImplemented = new Set<string>([
  "file.open",
  "file.close",
  "window.next",
  "window.previous",
  "image.zoom.in",
  "image.zoom.out",
  "image.zoom.reset",
]);

const launcherImplemented = new Set<string>([
  "file.open",
  "window.next",
  "window.previous",
]);

function AppInner() {
  const { toast } = useToast();

  const params = new URLSearchParams(window.location.search);
  const windowKind = params.get("window") === "viewer" ? "viewer" : "launcher";

  const menus = menuManifest as TopLevelMenu[];

  function handleDispatch(event: CommandDispatchEvent) {
    if (event.implemented) {
      toast({
        title: `Executed ${event.label}`,
        description: `Command: ${event.commandId}`,
      });
      return;
    }

    toast({
      title: `${event.label} is placeholder`,
      description: `Not implemented in clone phase 1 (${event.commandId}).`,
    });
  }

  if (windowKind === "viewer") {
    return (
      <>
        <ViewerWindowPage
          menus={menus}
          implementedCommands={viewerImplemented}
          onDispatch={handleDispatch}
        />
        <Toaster />
      </>
    );
  }

  return (
    <>
      <ControlWindowPage
        menus={menus}
        implementedCommands={launcherImplemented}
        onDispatch={handleDispatch}
      />
      <Toaster />
    </>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <AppInner />
    </ToastProvider>
  );
}
