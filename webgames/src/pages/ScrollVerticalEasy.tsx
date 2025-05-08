import { useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_ScrollVerticalEasy = "scroll-vertical-easy";
export const PASSWORD_ScrollVerticalEasy = "SCROLLMASTER2024EASY";

const ScrollVerticalEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ScrollVerticalEasy);
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
        threshold: 0.8, // 80% of the element must be visible
      }
    );

    if (lastBoxRef.current) {
      observer.observe(lastBoxRef.current);
    }

    return () => observer.disconnect();
  }, [recordSuccess]);

  // Generate content boxes - fewer boxes for easy mode
  const boxes = Array(20) // Reduced from 50 to 20
    .fill(null)
    .map((_, index) => (
      <div
        key={index}
        ref={index === 19 ? lastBoxRef : undefined} // Adjusted to 20 - 1
        className="h-[200px] m-5 rounded-lg flex items-center justify-center text-2xl text-gray-600"
        style={{
          backgroundColor: `hsl(${(index * 18) % 360}, 70%, 80%)`, // Adjusted hue step
        }}
      >
        Keep scrolling! {20 - index} boxes to go...
      </div>
    ));

  return (
    <div className="p-5">
      <h1>Scroll to the Bottom (Easy)</h1>
      <p>Keep scrolling down to reveal the secret password!</p>

      {boxes}

      {isLastBoxVisible && (
        <div className="p-5 bg-green-500 text-white rounded-lg text-center mt-5">
          Congratulations! You've reached the bottom!
          <div className="mt-2.5 text-2xl font-bold">
            Password: {PASSWORD_ScrollVerticalEasy}
          </div>
        </div>
      )}
    </div>
  );
};

export default ScrollVerticalEasy;
