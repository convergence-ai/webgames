import { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_SliderSymphonyEasy = "slider-symphony-easy";
export const PASSWORD_SliderSymphonyEasy = "SMOOTHANDEASY";

const SliderSymphonyEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_SliderSymphonyEasy);
  const [sliderValues, setSliderValues] = useState([50, 50]); // Reduced to 2 sliders
  const [targetPositions] = useState(() => {
    // Generate random target positions between 10 and 250 to keep within bounds
    return Array.from(
      { length: 2 }, // Reduced to 2 sliders
      () => Math.floor(Math.random() * 240) + 10
    );
  });
  const [isComplete, setIsComplete] = useState(false);

  // Convert slider value (0-100) to box position (10-250)
  const getBoxPosition = (value: number) => {
    return 10 + (value * 240) / 100; // This gives us a range of 10-250px
  };

  useEffect(() => {
    const aligned = sliderValues.every(
      (value, index) =>
        Math.abs(getBoxPosition(value) - targetPositions[index]) < 10
    );
    setIsComplete(aligned);
    if (aligned) {
      recordSuccess();
    }
  }, [sliderValues, targetPositions, recordSuccess]);

  const handleSliderChange = (index: number, value: number) => {
    setSliderValues((prev) => {
      const newValues = [...prev];
      newValues[index] = value;
      return newValues;
    });
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-b from-blue-50 to-indigo-50">
      <div className="w-full max-w-4xl px-8 py-6">
        <h1 className="text-3xl font-bold text-center text-indigo-900 mb-2">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-600">
            Slider Symphony (Easy)
          </span>
        </h1>

        {isComplete ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="p-8 bg-gradient-to-r from-green-100 to-emerald-100 text-emerald-700 rounded-xl shadow-lg border border-green-200 text-center">
              <div className="text-4xl mb-4">🎉</div>
              <div className="text-xl font-medium mb-4">
                Congratulations! You've aligned all the boxes!
              </div>
              <div className="font-mono bg-white/50 px-4 py-2 rounded-md inline-block">
                Password: {PASSWORD_SliderSymphonyEasy}
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Instructions */}
            <div className="mb-4 text-gray-700 bg-white/70 p-4 rounded-xl shadow-md backdrop-blur-sm">
              <p className="text-base font-medium">
                Align the colored boxes with their dashed outlines by adjusting
                the sliders. This version has fewer sliders.
              </p>
              <p className="text-sm mt-2 text-gray-600 flex items-center gap-2">
                <svg
                  className="w-4 h-4 text-blue-500"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                The boxes need to be very close to their targets to count as
                aligned!
              </p>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-center gap-8 bg-white/70 rounded-xl p-6 shadow-lg backdrop-blur-sm">
              {/* Sliders container */}
              <div className="flex gap-8">
                {sliderValues.map((value, index) => (
                  <div key={index} className="flex flex-col items-center">
                    <input
                      type="range"
                      value={value}
                      onChange={(e) =>
                        handleSliderChange(index, parseInt(e.target.value))
                      }
                      className="h-48 -rotate-180"
                      min="0"
                      max="100"
                      step="1"
                      style={{
                        writingMode: "vertical-lr",
                        background: `linear-gradient(to top, #60a5fa ${value}%, #e2e8f0 ${value}%)`,
                      }}
                    />
                  </div>
                ))}
              </div>

              {/* Boxes container */}
              <div className="relative w-80 h-[300px] flex items-center justify-center bg-white/30 rounded-lg">
                <div className="relative w-64 h-[260px]">
                  {sliderValues.map((value, index) => (
                    <div key={index} className="absolute w-full">
                      {/* Moving box */}
                      <div
                        className="w-10 h-10 rounded-lg shadow-lg absolute"
                        style={{
                          backgroundColor: isComplete ? "#4ade80" : "#60a5fa",
                          top: `${getBoxPosition(value)}px`,
                          left: `${index * 52}px`, // Spacing might need adjustment for fewer boxes
                        }}
                      />
                      {/* Target box */}
                      <div
                        className="w-10 h-10 rounded-lg border-2 border-dashed absolute"
                        style={{
                          borderColor: isComplete ? "#4ade80" : "#60a5fa",
                          top: `${targetPositions[index]}px`,
                          left: `${index * 52}px`, // Spacing might need adjustment
                          opacity: 0.6,
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default SliderSymphonyEasy;
