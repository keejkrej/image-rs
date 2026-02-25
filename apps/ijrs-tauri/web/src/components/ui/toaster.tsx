import { ToastProviderPrimitive, ToastViewport, Toast, ToastDescription, ToastTitle } from "@/components/ui/toast";
import { useToast } from "@/hooks/use-toast";

export function Toaster() {
  const { toasts, dismiss } = useToast();

  return (
    <ToastProviderPrimitive>
      {toasts.map((item) => (
        <Toast key={item.id} open onOpenChange={(open) => !open && dismiss(item.id)}>
          <div className="grid gap-1">
            <ToastTitle>{item.title}</ToastTitle>
            {item.description ? <ToastDescription>{item.description}</ToastDescription> : null}
          </div>
        </Toast>
      ))}
      <ToastViewport />
    </ToastProviderPrimitive>
  );
}
