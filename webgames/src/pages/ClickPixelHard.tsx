import React, { useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_ClickPixelHard = "PIXEL_HARD_2024";
export const TASK_ID_ClickPixelHard = "click-pixel-hard";

export default function ClickPixelHard() {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ClickPixelHard);
  const containerRef = useRef<HTMLDivElement>(null);
  const [password, setPassword] = useState("");

  const targetSize = 1; // Smaller target for hard mode

  // Ensure target is not too close to the edge, considering its small size
  const minOffset = 50;
  const targetPosition = {
    x:
      Math.floor(
        Math.random() * (window.innerWidth - 2 * minOffset - targetSize)
      ) + minOffset,
    y:
      Math.floor(
        Math.random() * (window.innerHeight - 2 * minOffset - targetSize)
      ) + minOffset,
  };

  const handleClick = (e: React.MouseEvent) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // For a 1x1 target, the click must be exactly on that pixel.
    const isWithinX =
      clickX >= targetPosition.x && clickX < targetPosition.x + targetSize;
    const isWithinY =
      clickY >= targetPosition.y && clickY < targetPosition.y + targetSize;

    if (isWithinX && isWithinY) {
      setPassword(PASSWORD_ClickPixelHard);
      recordSuccess();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-screen h-screen bg-gray-200 select-none cursor-crosshair" // Slightly different bg
      onClick={handleClick}
    >
      <div
        className="absolute bg-red-600" // Changed color for distinction
        style={{
          width: `${targetSize}px`,
          height: `${targetSize}px`,
          left: `${targetPosition.x}px`,
          top: `${targetPosition.y}px`,
        }}
      />

      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-white p-4 rounded-lg shadow-lg">
        <h1 className="text-xl font-bold mb-2">
          Pixel Perfect Challenge (Hard)
        </h1>
        <p className="text-gray-700 mb-2">
          Find and click the single {targetSize}x{targetSize} red pixel target
          hidden on the screen.
        </p>
        <p className="text-sm text-yellow-600">
          No hints will be provided for the target's location. Good luck!
        </p>
        {password && (
          <div className="mt-4 p-2 bg-green-100 rounded">
            <p className="text-green-800">
              Incredible! You hit the 1x1 target!
            </p>
            <p className="font-bold">Password: {password}</p>
          </div>
        )}
      </div>
    </div>
  );
}
