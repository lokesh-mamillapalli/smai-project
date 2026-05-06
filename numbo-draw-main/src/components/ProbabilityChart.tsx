import { MODEL_CONFIG } from "@/lib/model-config";

interface Props {
  probabilities: number[];
  predicted: number;
}

export function ProbabilityChart({ probabilities, predicted }: Props) {
  return (
    <div className="space-y-1.5">
      {probabilities.map((p, i) => {
        const pct = Math.round(p * 100);
        const isTop = i === predicted;
        return (
          <div key={i} className="flex items-center gap-3">
            <span
              className={`w-5 text-right font-mono text-sm ${
                isTop ? "font-bold text-primary" : "text-muted-foreground"
              }`}
            >
              {MODEL_CONFIG.labels[i]}
            </span>
            <div className="relative h-5 flex-1 overflow-hidden rounded-full bg-secondary">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isTop ? "bg-gradient-primary" : "bg-muted-foreground/30"
                }`}
                style={{ width: `${Math.max(pct, 1)}%` }}
              />
            </div>
            <span
              className={`w-12 text-right font-mono text-xs ${
                isTop ? "font-semibold text-foreground" : "text-muted-foreground"
              }`}
            >
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
