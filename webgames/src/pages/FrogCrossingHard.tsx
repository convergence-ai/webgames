import { useCallback, useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_FrogCrossingHard = "HOPPY_CROSSING_HARD";
export const TASK_ID_FrogCrossingHard = "frog-crossing-hard";

interface Position {
  x: number;
  y: number;
}

interface Car {
  x: number;
  y: number;
  direction: "left" | "right";
  speed: number;
}

const GRID_SIZE = 11; // Larger grid
const CAR_ROWS = [1, 2, 3, 5, 6, 7, 8]; // More car rows
const CARS_PER_ROW = 4; // More cars per row
const MOVE_INTERVAL = 300; // Faster car movement

export default function FrogCrossingHard() {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_FrogCrossingHard);
  const [frog, setFrog] = useState<Position>({
    x: Math.floor(GRID_SIZE / 2),
    y: GRID_SIZE - 1,
  });
  const [cars, setCars] = useState<Car[]>([]);
  const [gameOver, setGameOver] = useState(false);
  const [success, setSuccess] = useState(false);

  // Initialize cars
  useEffect(() => {
    const initialCars: Car[] = [];
    CAR_ROWS.forEach((row, index) => {
      for (let i = 0; i < CARS_PER_ROW; i++) {
        initialCars.push({
          x: Math.floor(Math.random() * GRID_SIZE),
          y: row,
          direction: index % 2 === 0 ? "left" : "right",
          speed: 1, // Speed can be adjusted per car or row for more difficulty
        });
      }
    });
    setCars(initialCars);
  }, []);

  // Move cars
  useEffect(() => {
    if (gameOver || success) return;

    const interval = setInterval(() => {
      setCars((prevCars) =>
        prevCars.map((car) => {
          let newX =
            car.direction === "left" ? car.x - car.speed : car.x + car.speed;

          if (newX < 0) newX = GRID_SIZE - 1;
          if (newX >= GRID_SIZE) newX = 0;

          return { ...car, x: newX };
        })
      );
    }, MOVE_INTERVAL);

    return () => clearInterval(interval);
  }, [gameOver, success]);

  // Check collisions and win condition
  useEffect(() => {
    const collision = cars.some((car) => car.x === frog.x && car.y === frog.y);

    if (collision) {
      setGameOver(true);
    }

    if (frog.y === 0) {
      setSuccess(true);
      recordSuccess();
    }
  }, [frog, cars, recordSuccess]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (gameOver || success) return;

      setFrog((prev) => {
        let newX = prev.x;
        let newY = prev.y;

        switch (e.key) {
          case "ArrowUp":
            newY = Math.max(0, prev.y - 1);
            break;
          case "ArrowDown":
            newY = Math.min(GRID_SIZE - 1, prev.y + 1);
            break;
          case "ArrowLeft":
            newX = Math.max(0, prev.x - 1);
            break;
          case "ArrowRight":
            newX = Math.min(GRID_SIZE - 1, prev.x + 1);
            break;
        }
        return { x: newX, y: newY };
      });
    },
    [gameOver, success]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-emerald-900 p-8">
      <div className="mb-6 text-2xl font-semibold">
        {gameOver ? (
          <div className="text-red-400">
            Game Over! Press R to restart (not implemented)
          </div>
        ) : success ? (
          <div className="text-emerald-400">
            Success! The password is: {PASSWORD_FrogCrossingHard}
          </div>
        ) : (
          <div className="text-emerald-300">
            Guide the frog home! Use arrow keys. (Hard)
          </div>
        )}
      </div>

      <div
        className="grid gap-0 bg-emerald-800 p-4 rounded-xl shadow-lg border-4 border-emerald-700"
        style={{
          gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`,
          width: `${GRID_SIZE * 35}px`, // Adjusted for more cells
          height: `${GRID_SIZE * 35}px`, // Adjusted for more cells
        }}
      >
        {Array.from({ length: GRID_SIZE * GRID_SIZE }).map((_, index) => {
          const x = index % GRID_SIZE;
          const y = Math.floor(index / GRID_SIZE);
          const isFrog = x === frog.x && y === frog.y;
          const car = cars.find((c) => c.x === x && c.y === y);

          return (
            <div
              key={index}
              className={`w-full h-full flex items-center justify-center border-[0.5px] border-emerald-700/30 ${
                y === 0
                  ? "bg-emerald-500"
                  : y === GRID_SIZE - 1
                  ? "bg-emerald-500"
                  : CAR_ROWS.includes(y)
                  ? "bg-slate-700"
                  : "bg-emerald-700"
              } min-h-[30px] min-w-[30px] text-xl`}
            >
              {isFrog ? (
                <span className="text-2xl transform hover:scale-110 transition-transform absolute">
                  🐸
                </span>
              ) : car ? (
                <span className="text-2xl transform hover:scale-105 transition-transform absolute">
                  {car.direction === "left" ? "🚙" : "🚗"}
                </span>
              ) : (
                <span className="opacity-0 select-none absolute">·</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
