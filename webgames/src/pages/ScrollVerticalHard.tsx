import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ScrollVerticalHard = "scroll-vertical-hard";
export const PASSWORD_ScrollVerticalHard = "SCROLLMASTER2024HARD";

const ScrollVerticalHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ScrollVerticalHard);
  const [isLastBoxVisible, setIsLastBoxVisible] = useState(false);
  const lastBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsLastBoxVisible(entry.isIntersecting);
        if (entry.isIntersecting) {
          recordSuccess();
        }
      },
      {
        threshold: 1.0, // 100% of the element must be visible
      }
    );

    if (lastBoxRef.current) {
      observer.observe(lastBoxRef.current);
    }

    return () => observer.disconnect();
  }, [recordSuccess]);

  // Generate content boxes - more boxes for hard mode
  const boxes = Array(100) // Increased from 50 to 100
    .fill(null)
    .map((_, index) => (
      <div
        key={index}
        ref={index === 99 ? lastBoxRef : undefined} // Adjusted to 100 - 1
        className="h-[200px] m-5 rounded-lg flex items-center justify-center text-2xl text-gray-600"
        style={{
          backgroundColor: `hsl(${(index * 3.6) % 360}, 70%, 80%)`, // Adjusted hue step
        }}
      >
        Keep scrolling! {100 - index} boxes to go...
      </div>
    ));

  return (
    <div className="p-5">
      <h1>Scroll to the Bottom (Hard)</h1>
      <p>
        Keep scrolling down to reveal the secret password! It's a long way
        down...
      </p>

      {boxes}

      {isLastBoxVisible && (
        <div className="p-5 bg-red-500 text-white rounded-lg text-center mt-5">
          Unbelievable! You actually made it to the bottom!
          <div className="mt-2.5 text-2xl font-bold">
            Password: {PASSWORD_ScrollVerticalHard}
          </div>
        </div>
      )}
    </div>
  );
};

export default ScrollVerticalHard;
