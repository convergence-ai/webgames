import React, { useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_MapPannerHard = "CARTOGRAPHERHARD2024";
export const TASK_ID_MapPannerHard = "map-panner-hard";

interface Position {
  x: number;
  y: number;
}

const MapPannerHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_MapPannerHard);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [startDrag, setStartDrag] = useState<Position>({ x: 0, y: 0 });
  const [startPos, setStartPos] = useState<Position>({ x: 0, y: 0 });
  const [isComplete, setIsComplete] = useState(false);

  // The treasure location is somewhere in the map
  const treasureLocation = { x: 2500, y: 2200 }; // Further for hard mode
  const treasureRadius = 100; // Smaller radius for hard mode

  // Calculate the viewport center position on the map
  const getViewportCenter = () => ({
    x: -position.x + window.innerWidth / 2,
    y: -position.y + window.innerHeight / 2,
  });

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setStartDrag({ x: e.clientX, y: e.clientY });
    setStartPos({ ...position });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const dx = e.clientX - startDrag.x;
    const dy = e.clientY - startDrag.y;

    setPosition({
      x: startPos.x + dx,
      y: startPos.y + dy,
    });

    // Check if we're near the treasure
    const viewportCenter = getViewportCenter();
    const distance = Math.sqrt(
      Math.pow(viewportCenter.x - treasureLocation.x, 2) +
        Math.pow(viewportCenter.y - treasureLocation.y, 2)
    );

    if (distance < treasureRadius && !isComplete) {
      setIsComplete(true);
      recordSuccess();
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Generate a grid of coordinates as map markers
  const generateMapMarkers = () => {
    const markers = [];
    for (let x = 0; x < 3000; x += 200) {
      // Increased map size
      for (let y = 0; y < 3000; y += 200) {
        // Increased map size
        markers.push(
          <div
            key={`${x}-${y}`}
            className="absolute text-gray-400 text-sm"
            style={{ left: x, top: y }}
          >
            {`${x},${y}`}
          </div>
        );
      }
    }
    return markers;
  };

  // Get the current viewport center for the scanning circle
  const viewportCenter = getViewportCenter();

  return (
    <div className="w-full h-screen overflow-hidden bg-gray-900 text-white">
      <div className="fixed top-4 left-4 z-10 bg-gray-800 p-4 rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold mb-2">Map Panner (Hard)</h1>
        <p className="text-gray-300">
          {isComplete
            ? "You found the hidden treasure! Impressive!"
            : "Pan this vast map to find the elusive treasure. It won't be easy... try the southeast quadrant if you dare."}
        </p>
        {isComplete && (
          <div className="mt-4 p-4 bg-green-800 rounded-lg">
            <p className="font-bold">Password: {PASSWORD_MapPannerHard}</p>
          </div>
        )}
      </div>

      <div
        ref={containerRef}
        className={`w-full h-full ${
          isDragging ? "cursor-grabbing" : "cursor-grab"
        }`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          className="relative"
          style={{
            transform: `translate(${position.x}px, ${position.y}px)`,
            width: "3000px", // Increased map size
            height: "3000px", // Increased map size
          }}
        >
          {/* Map grid background */}
          <div className="absolute inset-0 bg-gray-800">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)",
                backgroundSize: "100px 100px",
              }}
            />
          </div>

          {/* Map markers */}
          {generateMapMarkers()}

          {/* Scanning circle */}
          <div
            className="absolute border-2 border-red-400 rounded-full opacity-50" // Changed color for hard mode
            style={{
              left: viewportCenter.x,
              top: viewportCenter.y,
              width: treasureRadius * 2,
              height: treasureRadius * 2,
              transform: "translate(-50%, -50%)",
              transition: "all 0.1s ease-out",
            }}
          >
            <div className="absolute inset-0 bg-red-400 opacity-10 rounded-full" />
          </div>

          {/* Hidden treasure (only visible when found) */}
          {isComplete && (
            <div
              className="absolute animate-pulse"
              style={{
                left: treasureLocation.x,
                top: treasureLocation.y,
                transform: "translate(-50%, -50%)",
              }}
            >
              <span className="text-6xl">💎</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MapPannerHard;
