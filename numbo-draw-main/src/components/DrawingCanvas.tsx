import { useEffect, useRef, useState, useImperativeHandle, forwardRef } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Eraser } from "lucide-react";

export interface DrawingCanvasHandle {
  getCanvas: () => HTMLCanvasElement | null;
  clear: () => void;
  isEmpty: () => boolean;
}

interface Props {
  onChange?: () => void;
  guideTarget?: number | null;
}

const KANNADA_DIGITS = ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯'];

export const DrawingCanvas = forwardRef<DrawingCanvasHandle, Props>(({ onChange, guideTarget }, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastRef = useRef<{ x: number; y: number } | null>(null);
  const emptyRef = useRef(true);
  const [brush, setBrush] = useState(18);

  const reset = () => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, c.width, c.height);
    emptyRef.current = true;
    onChange?.();
  };

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    clear: reset,
    isEmpty: () => emptyRef.current,
  }));

  useEffect(() => {
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pos = (e: React.PointerEvent) => {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * c.width,
      y: ((e.clientY - r.top) / r.height) * c.height,
    };
  };

  const start = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drawingRef.current = true;
    lastRef.current = pos(e);
  };

  const move = (e: React.PointerEvent) => {
    if (!drawingRef.current) return;
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    const p = pos(e);
    const last = lastRef.current ?? p;
    ctx.strokeStyle = "black";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    lastRef.current = p;
    emptyRef.current = false;
    onChange?.();
  };

  const end = () => {
    drawingRef.current = false;
    lastRef.current = null;
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="relative rounded-2xl bg-gradient-soft p-3 shadow-soft">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          onPointerDown={start}
          onPointerMove={move}
          onPointerUp={end}
          onPointerLeave={end}
          className="aspect-square w-full touch-none rounded-xl bg-white shadow-inner"
          style={{ cursor: "crosshair" }}
        />
        {guideTarget !== undefined && guideTarget !== null && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <span className="select-none text-[12rem] font-bold text-foreground/10">
              {KANNADA_DIGITS[guideTarget]}
            </span>
          </div>
        )}
      </div>
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <Label className="mb-2 flex justify-between text-xs text-muted-foreground">
            <span>Brush size</span>
            <span className="font-mono">{brush}px</span>
          </Label>
          <Slider
            value={[brush]}
            onValueChange={(v) => setBrush(v[0])}
            min={4}
            max={40}
            step={1}
          />
        </div>
        <Button variant="outline" size="sm" onClick={reset}>
          <Eraser className="mr-1.5 h-4 w-4" />
          Clear
        </Button>
      </div>
    </div>
  );
});

DrawingCanvas.displayName = "DrawingCanvas";
