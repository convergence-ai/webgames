import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_BullseyeEasy = "bullseye-easy";
export const PASSWORD_BullseyeEasy = "BullseyeEasyPeasy2024";

const BullseyeEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_BullseyeEasy);
  const [position, setPosition] = useState({ x: 0, y: 50 });
  const [direction, setDirection] = useState(1);
  const [hits, setHits] = useState(0);
  const [speed, setSpeed] = useState(0.2); // Slower initial speed
  const [isComplete, setIsComplete] = useState(false);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    const animate = () => {
      setPosition((prev) => {
        let newX = prev.x + speed * direction;
        let newDirection = direction;

        if (newX >= 95 || newX <= 5) {
          newDirection *= -1;
          setDirection(newDirection);
          newX = newDirection > 0 ? 5 : 95;
        }

        return {
          x: newX,
          y: prev.y,
        };
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [direction, speed]);

  const handleClick = () => {
    if (isComplete) {
      recordSuccess();
      return;
    }

    setHits((prev) => {
      const newHits = prev + 1;
      if (newHits >= 1) {
        setIsComplete(true);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        return newHits;
      }
      setSpeed((prevSpeed) => prevSpeed * 1.05); // Slower speed increase
      return newHits;
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-100 to-green-100 p-4">
      <div className="max-w-4xl w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-blue-600">
          Bullseye Challenge (Easy)
        </h1>

        <div className="text-center mb-6">
          <p className="text-lg mb-2">Hits: {hits}/1</p>
        </div>

        {isComplete ? (
          <div className="text-center p-4 bg-green-100 text-green-700 rounded-lg">
            Congratulations! You've completed the easy challenge! 🎯 The secret
            password is: {PASSWORD_BullseyeEasy}
          </div>
        ) : (
          <div className="relative h-80 bg-gray-100 rounded-lg overflow-hidden">
            <button
              onClick={handleClick}
              style={{
                left: `${position.x}%`,
                top: `${position.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              className="absolute w-12 h-12 rounded-full bg-red-500 hover:bg-red-600 focus:outline-none transition-colors duration-200 flex items-center justify-center"
            >
              <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center">
                <div className="w-4 h-4 rounded-full bg-red-500" />
              </div>
            </button>
          </div>
        )}

        <p className="text-sm text-center text-gray-600 mt-4">
          Click the moving target one time to win. It gets slightly faster with
          each hit!
        </p>
      </div>
    </div>
  );
};

export default BullseyeEasy;
