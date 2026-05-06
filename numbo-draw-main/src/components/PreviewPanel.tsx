import { useEffect, useRef } from "react";

interface Props {
  canvas: HTMLCanvasElement | null;
}

export function PreviewPanel({ canvas }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    host.innerHTML = "";
    if (canvas) {
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.imageRendering = "pixelated";
      canvas.style.borderRadius = "0.5rem";
      host.appendChild(canvas);
    }
  }, [canvas]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="aspect-square w-24 overflow-hidden rounded-lg bg-muted ring-1 ring-border">
        <div ref={hostRef} className="h-full w-full" />
      </div>
      <span className="text-xs text-muted-foreground">28×28 input</span>
    </div>
  );
}
