import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_BullseyeHard = "bullseye-hard";
export const PASSWORD_BullseyeHard = "BullseyeMaster2024";

const BullseyeHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_BullseyeHard);
  const [position, setPosition] = useState({ x: 0, y: 50 });
  const [direction, setDirection] = useState(1);
  const [hits, setHits] = useState(0);
  const [speed, setSpeed] = useState(0.5); // Faster initial speed
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
      if (newHits >= 4) {
        // 4 hits required
        setIsComplete(true);
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
        return newHits;
      }
      setSpeed((prevSpeed) => prevSpeed * 1.25); // Faster speed increase
      return newHits;
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-red-100 to-yellow-100 p-4">
      <div className="max-w-4xl w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-red-600">
          Bullseye Challenge (Hard)
        </h1>

        <div className="text-center mb-6">
          <p className="text-lg mb-2">Hits: {hits}/4</p>
        </div>

        {isComplete ? (
          <div className="text-center p-4 bg-green-100 text-green-700 rounded-lg">
            Incredible! You've mastered the hard challenge! 🎯 The secret
            password is: {PASSWORD_BullseyeHard}
          </div>
        ) : (
          <div className="relative h-80 bg-gray-200 rounded-lg overflow-hidden">
            <button
              onClick={handleClick}
              style={{
                left: `${position.x}%`,
                top: `${position.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              className="absolute w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-700 focus:outline-none transition-colors duration-100 flex items-center justify-center"
            >
              <div className="w-6 h-6 rounded-full bg-white flex items-center justify-center">
                <div className="w-3 h-3 rounded-full bg-purple-600" />
              </div>
            </button>
          </div>
        )}

        <p className="text-sm text-center text-gray-700 mt-4">
          Click the rapidly moving target four times to win. It gets much faster
          with each hit!
        </p>
      </div>
    </div>
  );
};

export default BullseyeHard;
