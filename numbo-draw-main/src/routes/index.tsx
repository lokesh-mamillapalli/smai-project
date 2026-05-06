import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import {
  Brain,
  Sparkles,
  Upload,
  Pencil,
  Loader2,
  RotateCcw,
  Target,
  CheckCircle2,
  XCircle,
  BarChart3,
  Image as ImageIcon,
} from "lucide-react";
import confetti from "canvas-confetti";
import { DrawingCanvas, type DrawingCanvasHandle } from "@/components/DrawingCanvas";
import { ProbabilityChart } from "@/components/ProbabilityChart";
import { PreviewPanel } from "@/components/PreviewPanel";
import { loadModel, predict, type PredictionResult } from "@/lib/model-loader";
import { preprocess } from "@/lib/preprocessing";
import { MODEL_CONFIG } from "@/lib/model-config";

export const Route = createFileRoute("/")({
  component: HomePage,
  head: () => ({
    meta: [
      { title: "Handwritten Digit Recognizer" },
      {
        name: "description",
        content:
          "Draw or upload a handwritten digit and get instant predictions powered by an in-browser neural network.",
      },
    ],
  }),
});

function HomePage() {
  const canvasRef = useRef<DrawingCanvasHandle>(null);
  const [uploadedImg, setUploadedImg] = useState<HTMLImageElement | null>(null);
  const [uploadedUrl, setUploadedUrl] = useState<string | null>(null);
  const [tab, setTab] = useState<"draw" | "upload">("draw");
  const [isDragging, setIsDragging] = useState(false);

  const [modelReady, setModelReady] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [previewCanvas, setPreviewCanvas] = useState<HTMLCanvasElement | null>(null);

  // Practice mode
  const [practiceTarget, setPracticeTarget] = useState<number | null>(null);
  const [practiceFeedback, setPracticeFeedback] = useState<"correct" | "wrong" | null>(null);

  // Stats
  const [predictionCount, setPredictionCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);

  useEffect(() => {
    let mounted = true;
    loadModel()
      .then(() => {
        if (mounted) {
          setModelReady(true);
          setModelLoading(false);
        }
      })
      .catch((e) => {
        console.error(e);
        if (mounted) {
          setModelLoading(false);
          toast.error("Failed to load model. Check your network or model URL.");
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  const processFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file (JPG or PNG).");
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setUploadedImg(img);
      setUploadedUrl(url);
    };
    img.onerror = () => toast.error("Could not read that image.");
    img.src = url;
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    processFile(e.target.files?.[0]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    processFile(e.dataTransfer.files?.[0]);
  };

  const runPrediction = async () => {
    if (!modelReady) {
      toast.error("Model isn't ready yet.");
      return;
    }
    let source: HTMLCanvasElement | HTMLImageElement | null = null;
    if (tab === "draw") {
      if (canvasRef.current?.isEmpty()) {
        toast.error("Draw a digit first!");
        return;
      }
      source = canvasRef.current?.getCanvas() ?? null;
    } else {
      if (!uploadedImg) {
        toast.error("Upload an image first!");
        return;
      }
      source = uploadedImg;
    }
    if (!source) return;

    setPredicting(true);
    setResult(null);
    try {
      const { data, previewCanvas } = preprocess(source);
      setPreviewCanvas(previewCanvas);
      const res = await predict(data);
      setResult(res);
      setPredictionCount((c) => c + 1);

      if (res.confidence > 0.97) {
        confetti({
          particleCount: 150,
          spread: 80,
          origin: { y: 0.6 }
        });
      }

      if (practiceTarget !== null) {
        const isCorrect = res.predicted === practiceTarget;
        setPracticeFeedback(isCorrect ? "correct" : "wrong");
        if (isCorrect) setCorrectCount((c) => c + 1);
      }
    } catch (e) {
      console.error(e);
      toast.error("Prediction failed. See console for details.");
    } finally {
      setPredicting(false);
    }
  };

  const resetAll = () => {
    canvasRef.current?.clear();
    setResult(null);
    setPreviewCanvas(null);
    setPracticeFeedback(null);
    if (uploadedUrl) URL.revokeObjectURL(uploadedUrl);
    setUploadedImg(null);
    setUploadedUrl(null);
  };

  const newPracticeRound = () => {
    setPracticeTarget(Math.floor(Math.random() * 10));
    setPracticeFeedback(null);
    setResult(null);
    canvasRef.current?.clear();
    setTab("draw");
  };

  const quitPractice = () => {
    setPracticeTarget(null);
    setPracticeFeedback(null);
    setResult(null);
    canvasRef.current?.clear();
  };

  return (
    <div className="min-h-screen px-4 py-8 sm:py-12">
      <Toaster position="top-center" richColors />

      <div className="mx-auto max-w-6xl">
        {/* Header */}
        <header className="mb-10 text-center animate-slide-up">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-card px-4 py-1.5 text-xs font-medium text-muted-foreground shadow-soft ring-1 ring-border">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Powered by TensorFlow.js · runs entirely in your browser
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gradient sm:text-5xl">
            Handwritten Digit Recognizer
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base text-muted-foreground sm:text-lg">
            Draw or upload a digit and get instant prediction
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-5">
          {/* INPUT CARD */}
          <Card className="lg:col-span-3 overflow-hidden border-border/60 p-6 shadow-elegant animate-slide-up" style={{ animationDelay: "100ms", animationFillMode: "both" }}>
            <Tabs value={tab} onValueChange={(v) => setTab(v as "draw" | "upload")}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="draw">
                  <Pencil className="mr-2 h-4 w-4" /> Draw
                </TabsTrigger>
                <TabsTrigger value="upload">
                  <Upload className="mr-2 h-4 w-4" /> Upload
                </TabsTrigger>
              </TabsList>

              <TabsContent value="draw" className="mt-5">
                <DrawingCanvas ref={canvasRef} onChange={() => setPracticeFeedback(null)} guideTarget={practiceTarget} />
              </TabsContent>

              <TabsContent value="upload" className="mt-5">
                <div className="flex flex-col items-center gap-4">
                  <label
                    className={`group flex aspect-square w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed ${isDragging ? "border-primary bg-primary/5" : "border-border bg-gradient-soft"} p-6 transition hover:border-primary hover:shadow-soft`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                    {uploadedUrl ? (
                      <img
                        src={uploadedUrl}
                        alt="Uploaded digit"
                        className="max-h-full max-w-full rounded-xl object-contain shadow-soft"
                      />
                    ) : (
                      <>
                        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-card shadow-soft transition group-hover:scale-110">
                          <ImageIcon className="h-6 w-6 text-primary" />
                        </div>
                        <div className="text-center">
                          <p className="font-medium text-foreground">Click to upload an image</p>
                          <p className="mt-1 text-xs text-muted-foreground">JPG or PNG</p>
                        </div>
                      </>
                    )}
                    <input
                      type="file"
                      accept="image/png,image/jpeg"
                      className="hidden"
                      onChange={handleUpload}
                    />
                  </label>
                </div>
              </TabsContent>
            </Tabs>

            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <Button
                onClick={runPrediction}
                disabled={!modelReady || predicting}
                className="flex-1 bg-gradient-primary text-primary-foreground shadow-soft transition hover:shadow-glow"
                size="lg"
              >
                {predicting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Predicting…
                  </>
                ) : modelLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading model…
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Predict digit
                  </>
                )}
              </Button>
              <Button onClick={resetAll} variant="outline" size="lg">
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset
              </Button>
            </div>
          </Card>

          {/* RESULT CARD */}
          <Card className="lg:col-span-2 flex flex-col gap-5 border-border/60 p-6 shadow-elegant animate-slide-up" style={{ animationDelay: "200ms", animationFillMode: "both" }}>
            {practiceTarget !== null && (
              <div className="rounded-xl bg-gradient-soft p-4 ring-1 ring-border animate-pop-in">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Target className="h-4 w-4 text-primary" />
                    Practice — draw this digit:
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={quitPractice}>
                      Quit
                    </Button>
                    <Button variant="ghost" size="sm" onClick={newPracticeRound}>
                      New
                    </Button>
                  </div>
                </div>
                <div className="mt-2 flex items-center gap-4">
                  <div className="text-5xl font-bold text-gradient">{practiceTarget}</div>
                  {practiceFeedback === "correct" && (
                    <Badge className="bg-success/15 text-success border-success/30">
                      <CheckCircle2 className="mr-1 h-3.5 w-3.5" /> Correct!
                    </Badge>
                  )}
                  {practiceFeedback === "wrong" && (
                    <Badge variant="destructive">
                      <XCircle className="mr-1 h-3.5 w-3.5" /> Try again
                    </Badge>
                  )}
                </div>
              </div>
            )}

            {result ? (
              <div className="flex flex-col gap-5 animate-pop-in">
                <div className="flex items-center gap-5">
                  <div className="flex h-28 w-28 shrink-0 items-center justify-center rounded-2xl bg-gradient-primary text-7xl font-bold text-primary-foreground shadow-glow animate-pulse-glow">
                    {result.label}
                  </div>
                  <div className="flex-1">
                    <div className="text-sm text-muted-foreground">Predicted digit</div>
                    <div className="mt-1 text-2xl font-bold text-foreground">
                      {Math.round(result.confidence * 100)}%
                    </div>
                    <div className="text-xs text-muted-foreground">confidence</div>
                  </div>
                  {previewCanvas && <PreviewPanel canvas={previewCanvas} />}
                </div>

                <div>
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                    <BarChart3 className="h-4 w-4 text-primary" />
                    Probability distribution
                  </div>
                  <ProbabilityChart
                    probabilities={result.probabilities}
                    predicted={result.predicted}
                  />
                </div>
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-3 py-10 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-secondary">
                  <Brain className="h-7 w-7 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">
                  {modelLoading
                    ? "Loading the neural network…"
                    : "Draw or upload a digit, then click Predict."}
                </p>
              </div>
            )}
          </Card>

          {/* PRACTICE MODE CARD */}
          <Card className="lg:col-span-2 border-border/60 p-6 shadow-soft animate-slide-up" style={{ animationDelay: "300ms", animationFillMode: "both" }}>
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-primary text-primary-foreground">
                <Target className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">Practice mode</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Get a random digit and see if you can fool the model.
                </p>
              </div>
            </div>
            <Button onClick={newPracticeRound} variant="outline" className="mt-4 w-full">
              <Sparkles className="mr-2 h-4 w-4" />
              {practiceTarget === null ? "Start practice" : "New round"}
            </Button>
          </Card>

          {/* MODEL INFO CARD */}
          <Card className="lg:col-span-2 border-border/60 p-6 shadow-soft animate-slide-up" style={{ animationDelay: "400ms", animationFillMode: "both" }}>
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-primary text-primary-foreground">
                <Brain className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">Model info</h3>
                <dl className="mt-3 space-y-1.5 text-sm">
                  <Row k="Model" v={MODEL_CONFIG.name} />
                  <Row
                    k="Input shape"
                    v={`${MODEL_CONFIG.inputShape[0]}×${MODEL_CONFIG.inputShape[1]}×${MODEL_CONFIG.inputShape[2]}`}
                  />
                  <Row k="Classes" v={`${MODEL_CONFIG.labels.length}`} />
                  <Row k="Accuracy" v={MODEL_CONFIG.accuracy} />
                  <Row k="Status" v={modelReady ? "Ready" : modelLoading ? "Loading" : "Error"} />
                </dl>
              </div>
            </div>
          </Card>

          {/* STATS */}
          <Card className="lg:col-span-1 border-border/60 p-6 shadow-soft animate-slide-up" style={{ animationDelay: "500ms", animationFillMode: "both" }}>
            <h3 className="text-sm font-medium text-muted-foreground">Session stats</h3>
            <div className="mt-3 space-y-3">
              <div>
                <div className="text-3xl font-bold text-gradient">{predictionCount}</div>
                <div className="text-xs text-muted-foreground">predictions</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-foreground">{correctCount}</div>
                <div className="text-xs text-muted-foreground">practice wins</div>
              </div>
            </div>
          </Card>
        </div>

        <footer className="mt-10 text-center text-xs text-muted-foreground">
          Swap the model: convert your <code className="font-mono">.h5</code> with{" "}
          <code className="font-mono">tensorflowjs_converter</code>, drop into{" "}
          <code className="font-mono">public/model/</code>, and update{" "}
          <code className="font-mono">src/lib/model-config.ts</code>.
        </footer>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className="truncate font-medium text-foreground">{v}</dd>
    </div>
  );
}
