import React, { useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_ClickPixelEasy = "PIXEL_EASY_2024";
export const TASK_ID_ClickPixelEasy = "click-pixel-easy";

export default function ClickPixelEasy() {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ClickPixelEasy);
  const containerRef = useRef<HTMLDivElement>(null);
  const [password, setPassword] = useState("");

  const targetSize = 15; // Larger target for easy mode

  const targetPosition = {
    x:
      Math.floor(Math.random() * (window.innerWidth - 2 * 100 - targetSize)) +
      100,
    y:
      Math.floor(Math.random() * (window.innerHeight - 2 * 100 - targetSize)) +
      100,
  };

  const [lastClick, setLastClick] = useState<{ x: number; y: number } | null>(
    null
  );

  const handleClick = (e: React.MouseEvent) => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    setLastClick({ x: clickX, y: clickY });

    const isWithinX =
      clickX >= targetPosition.x && clickX <= targetPosition.x + targetSize;
    const isWithinY =
      clickY >= targetPosition.y && clickY <= targetPosition.y + targetSize;

    if (isWithinX && isWithinY) {
      setPassword(PASSWORD_ClickPixelEasy);
      recordSuccess();
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-screen h-screen bg-gray-100 select-none cursor-crosshair"
      onClick={handleClick}
    >
      <div
        className="absolute bg-blue-500" // Changed color for distinction
        style={{
          width: `${targetSize}px`,
          height: `${targetSize}px`,
          left: `${targetPosition.x}px`,
          top: `${targetPosition.y}px`,
        }}
      />

      <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-white p-4 rounded-lg shadow-lg">
        <h1 className="text-xl font-bold mb-2">
          Pixel Perfect Challenge (Easy)
        </h1>
        <p className="text-gray-600 mb-2">
          Click anywhere within the {targetSize}x{targetSize} blue square target
          to complete the challenge.
        </p>
        <div className="text-sm text-gray-500">
          <p>
            Target position: ({targetPosition.x}, {targetPosition.y})
          </p>
          {lastClick && (
            <p className="mt-1">
              Last click: ({lastClick.x}, {lastClick.y}) - Distance from target
              corner: ({Math.abs(lastClick.x - targetPosition.x).toFixed(1)},{" "}
              {Math.abs(lastClick.y - targetPosition.y).toFixed(1)}) pixels
            </p>
          )}
        </div>
        {password && (
          <div className="mt-4 p-2 bg-green-100 rounded">
            <p className="text-green-800">
              Congratulations! You found the target!
            </p>
            <p className="font-bold">Password: {password}</p>
          </div>
        )}
      </div>
    </div>
  );
}
