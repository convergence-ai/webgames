import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface Circle {
  x: number;
  y: number;
  radius: number;
}

interface TargetZone {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const PASSWORD_CanvasCatchHard = "HardCircleMaster2024";
export const TASK_ID_CanvasCatchHard = "canvas-catch-hard";

const CanvasCatchHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_CanvasCatchHard);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [circle, setCircle] = useState<Circle>({ x: 100, y: 100, radius: 10 }); // Smaller circle
  const targetZone: TargetZone = { x: 450, y: 250, width: 50, height: 50 }; // Smaller target zone

  const draw = (ctx: CanvasRenderingContext2D) => {
    // Clear canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Draw target zone
    ctx.fillStyle = completed ? "#4CAF50" : "#FFE0B2";
    ctx.fillRect(
      targetZone.x,
      targetZone.y,
      targetZone.width,
      targetZone.height
    );
    ctx.strokeStyle = "#FF9800";
    ctx.strokeRect(
      targetZone.x,
      targetZone.y,
      targetZone.width,
      targetZone.height
    );

    // Draw circle
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#2196F3";
    ctx.fill();
    ctx.strokeStyle = "#1976D2";
    ctx.stroke();
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    canvas.width = 600;
    canvas.height = 400;

    draw(ctx);
  }, [circle, completed]);

  const isInTargetZone = () => {
    return (
      circle.x > targetZone.x &&
      circle.x < targetZone.x + targetZone.width &&
      circle.y > targetZone.y &&
      circle.y < targetZone.y + targetZone.height
    );
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if click is inside circle
    const distance = Math.sqrt(
      Math.pow(x - circle.x, 2) + Math.pow(y - circle.y, 2)
    );

    if (distance < circle.radius) {
      setIsDragging(true);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.max(
      circle.radius,
      Math.min(e.clientX - rect.left, canvas.width - circle.radius)
    );
    const y = Math.max(
      circle.radius,
      Math.min(e.clientY - rect.top, canvas.height - circle.radius)
    );

    setCircle((prev) => ({ ...prev, x, y }));
  };

  const handleMouseUp = () => {
    if (isDragging && isInTargetZone()) {
      setCompleted(true);
      recordSuccess();
    }
    setIsDragging(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-4">Canvas Catch (Hard)</h1>
      {completed ? (
        <div className="mt-6 p-8 bg-green-100 rounded-lg border border-green-200 text-center">
          <p className="text-xl text-green-800 font-bold mb-2">
            Congratulations!
          </p>
          <p className="text-green-800">
            You've successfully completed the hard challenge!
          </p>
          <p className="text-2xl font-bold text-green-900 mt-4">
            Secret Password: {PASSWORD_CanvasCatchHard}
          </p>
        </div>
      ) : (
        <>
          <p className="mb-4">
            Drag the blue circle into the orange box. This one is much harder!
          </p>
          <canvas
            ref={canvasRef}
            className="border border-gray-300 rounded-lg shadow-lg"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
        </>
      )}
    </div>
  );
};

export default CanvasCatchHard;
