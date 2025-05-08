import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
interface Sheep {
  x: number;
  y: number;
  vx: number;
  vy: number;
  id: number;
}

interface Pen {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Game dimensions
const GAME_WIDTH = 800;
const GAME_HEIGHT = 600;

export const TASK_ID_HerdingEasy = "herding-easy";
export const PASSWORD_HerdingEasy = "HerdTooEasy";

const HerdingEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_HerdingEasy);
  const [sheep, setSheep] = useState<Sheep[]>([]);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [score, setScore] = useState(0);
  const [hasWon, setHasWon] = useState(false);
  const [scale, setScale] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<number>();
  const pen: Pen = {
    x: 100,
    y: 100,
    width: 150,
    height: 150,
  };

  // Calculate and update scale based on window size
  useEffect(() => {
    const updateScale = () => {
      if (containerRef.current?.parentElement) {
        const parent = containerRef.current.parentElement;
        const maxWidth = parent.clientWidth;
        const maxHeight = window.innerHeight - 150; // Account for header elements
        const scaleX = maxWidth / GAME_WIDTH;
        const scaleY = maxHeight / GAME_HEIGHT;
        setScale(Math.min(scaleX, scaleY)); // Allow scaling to fit available space
      }
    };

    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, []);

  // Initialize sheep positions
  useEffect(() => {
    const initialSheep: Sheep[] = Array.from({ length: 2 }, (_, i) => ({
      // Easy: 2 sheep
      x: Math.random() * (GAME_WIDTH * 0.3) + GAME_WIDTH * 0.6, // Start on the right side
      y: Math.random() * (GAME_HEIGHT * 0.8) + GAME_HEIGHT * 0.1,
      vx: 0,
      vy: 0,
      id: i,
    }));
    setSheep(initialSheep);
  }, []);

  // Handle mouse movement
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setMousePos({
          x: (e.clientX - rect.left) / scale,
          y: (e.clientY - rect.top) / scale,
        });
      }
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener("mousemove", handleMouseMove);
      return () => container.removeEventListener("mousemove", handleMouseMove);
    }
  }, [scale]);

  // Animation loop
  useEffect(() => {
    const updateSheep = () => {
      setSheep((prevSheep) => {
        const newSheep = prevSheep.map((sheep) => {
          // Calculate distance from mouse
          const dx = sheep.x - mousePos.x;
          const dy = sheep.y - mousePos.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Apply repulsion force if mouse is close
          const repulsionRadius = 100;
          let ax = 0;
          let ay = 0;

          if (distance < repulsionRadius) {
            const force = (1 - distance / repulsionRadius) * 0.5;
            ax += (dx / distance) * force;
            ay += (dy / distance) * force;
          }

          // Update velocity with damping
          const damping = 0.95;
          const newVx = (sheep.vx + ax) * damping;
          const newVy = (sheep.vy + ay) * damping;

          // Update position with boundary checking
          const margin = 20; // Keep sheep slightly inside the borders
          const newX = Math.max(
            margin,
            Math.min(GAME_WIDTH - margin, sheep.x + newVx)
          );
          const newY = Math.max(
            margin,
            Math.min(GAME_HEIGHT - margin, sheep.y + newVy)
          );

          // If sheep hit boundaries, reverse velocity
          const vx =
            newX === margin || newX === GAME_WIDTH - margin
              ? -newVx * 0.5
              : newVx;
          const vy =
            newY === margin || newY === GAME_HEIGHT - margin
              ? -newVy * 0.5
              : newVy;

          return {
            ...sheep,
            x: newX,
            y: newY,
            vx,
            vy,
          };
        });

        // Check if sheep are in the pen
        const sheepInPen = newSheep.filter(
          (s) =>
            s.x > pen.x &&
            s.x < pen.x + pen.width &&
            s.y > pen.y &&
            s.y < pen.y + pen.height
        ).length;

        setScore(sheepInPen);

        // Check for victory
        if (sheepInPen === newSheep.length && newSheep.length > 0) {
          setHasWon(true);
        }

        return newSheep;
      });

      frameRef.current = requestAnimationFrame(updateSheep);
    };

    frameRef.current = requestAnimationFrame(updateSheep);
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [mousePos, pen.x, pen.y, pen.width, pen.height]);

  // Add new effect to handle recording success once
  useEffect(() => {
    if (hasWon) {
      recordSuccess();
    }
  }, [hasWon, recordSuccess]);

  return (
    <div className="w-full h-screen flex flex-col items-center bg-green-100">
      <div className="text-lg text-gray-700 mt-4">
        Use your cursor to herd the sheep into the pen!
      </div>
      <div className="text-2xl mb-2">
        Sheep in pen: {score} / {sheep.length}
      </div>
      {hasWon ? (
        <div className="text-center">
          <div className="text-4xl mb-4 text-green-600">
            🎉 Congratulations! 🎉
          </div>
          <div className="text-2xl mb-8">All sheep are safely in the pen!</div>
          <div className="text-xl bg-yellow-100 p-4 rounded-lg border-2 border-yellow-400">
            The secret password is:{" "}
            <span className="font-bold">{PASSWORD_HerdingEasy}</span>
          </div>
        </div>
      ) : (
        <div className="flex-1 w-full relative overflow-hidden flex justify-center items-center">
          <div
            ref={containerRef}
            className="relative border-4 border-green-800 bg-green-200"
            style={{
              width: GAME_WIDTH,
              height: GAME_HEIGHT,
              transform: `scale(${scale})`,
              transformOrigin: "center",
            }}
          >
            {/* Pen */}
            <div
              className="absolute border-4 border-brown-600 bg-yellow-100"
              style={{
                left: pen.x,
                top: pen.y,
                width: pen.width,
                height: pen.height,
              }}
            />

            {/* Sheep */}
            {sheep.map((sheep) => (
              <div
                key={sheep.id}
                className="absolute text-4xl"
                style={{
                  left: sheep.x,
                  top: sheep.y,
                  transform: "translate(-50%, -50%)",
                }}
              >
                🐑
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HerdingEasy;
