import React, { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_PopupChaosEasy = "popup-chaos-easy";
export const PASSWORD_PopupChaosEasy = "PopupSlayerEasy2024";

interface Popup {
  id: number;
  x: number;
  y: number;
  title: string;
  content: string;
  isDragging: boolean;
  offsetX: number;
  offsetY: number;
  cta: {
    text: string;
    className: string;
  };
}

const BASE_Z_INDEX = 100; // Base z-index for popups

const titles = [
  "Important Update!",
  "Special Offer!",
  "Breaking News!",
  "Don't Miss Out!",
  "Act Now!",
  "Limited Time!",
  "System Warning!",
  "Security Alert!",
  "Update Required!",
];
const contents = [
  "Click here to claim your prize! Our records show you've been selected for a special reward. Don't let this opportunity pass you by!",
  "You're the 1,000,000th visitor! We've been trying to reach you about your computer's extended warranty.",
  "Your computer needs attention! Critical system updates are available. Click here to optimize your system now!",
  "Download more RAM now! Our advanced technology allows you to instantly upgrade your computer's memory. Try it now!",
  "Hot singles in your area! They're waiting to meet someone just like you. Don't keep them waiting any longer!",
  "You've won a free iPhone! Plus a free tablet, laptop, and smart TV. Click now to claim your prizes!",
  "Warning: Your system is running low on resources! Click here for an instant performance boost.",
  "🎉 Congratulations! You've been selected for our exclusive offer. Limited time only - act fast!",
  "⚠️ System Alert: Multiple issues detected! Click here for an instant fix to all your problems.",
];
const ctas = [
  { text: "Learn More", className: "bg-blue-500 hover:bg-blue-600" },
  { text: "Get Started", className: "bg-indigo-500 hover:bg-indigo-600" },
  {
    text: "Yes, I'm Interested!",
    className: "bg-pink-500 hover:bg-pink-600",
  },
  { text: "Install Now", className: "bg-orange-500 hover:bg-orange-600" },
  { text: "Continue", className: "bg-teal-500 hover:bg-teal-600" },
  { text: "Accept Offer", className: "bg-cyan-500 hover:bg-cyan-600" },
  {
    text: "Close all popups",
    className: "bg-green-500 hover:bg-green-600",
  },
];

function createPopup(x?: number, y?: number): Popup {
  const randomCta = ctas[Math.floor(Math.random() * ctas.length)];

  return {
    id: Date.now() + Math.random(),
    x: x ?? Math.random() * (window.innerWidth - 400),
    y: y ?? Math.random() * (window.innerHeight - 300),
    title: titles[Math.floor(Math.random() * titles.length)],
    content: contents[Math.floor(Math.random() * contents.length)],
    isDragging: false,
    offsetX: 0,
    offsetY: 0,
    cta: randomCta,
  };
}

function makePopups(): Popup[] {
  const centerX = window.innerWidth / 2 - 200;
  const centerY = window.innerHeight / 2 - 150;

  // Reduced number of popups for Easy mode
  const positions = [
    [centerX - Math.random() * 100, centerY - Math.random() * 100],
    [centerX + Math.random() * 100, centerY + Math.random() * 100],
  ];

  const boundedPositions = positions.map(([x, y]) => [
    Math.max(0, Math.min(x, window.innerWidth - 400)),
    Math.max(0, Math.min(y, window.innerHeight - 300)),
  ]);

  return boundedPositions.map(([x, y]) => createPopup(x, y));
}

const PopupChaosEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_PopupChaosEasy);
  const [popups, setPopups] = useState<Popup[]>([]);
  const isComplete = popups.length === 0;

  // Initialize popups on mount
  useEffect(() => {
    setPopups(makePopups());
  }, []);

  useEffect(() => {
    if (isComplete && popups !== null && popups.length === 0) {
      // Check if popups was ever initialized
      const initialPopupCount = makePopups().length; // Rough check, assumes makePopups is deterministic without args for count
      if (initialPopupCount > 0) {
        // Only record success if there were popups to close
        recordSuccess();
      }
    }
  }, [isComplete, recordSuccess, popups]);

  const bringToFront = (id: number) => {
    setPopups((prev) => {
      if (!prev) return [];
      const popupIndex = prev.findIndex((p) => p.id === id);
      if (popupIndex === -1) return prev;

      const newPopups = [...prev];
      const [popup] = newPopups.splice(popupIndex, 1);
      return [...newPopups, popup];
    });
  };

  const handleMouseDown = (e: React.MouseEvent, id: number) => {
    e.preventDefault(); // Prevent text selection while dragging
    const popupEl = e.currentTarget.closest(".fixed") as HTMLElement;
    if (!popupEl) return;
    const popupRect = popupEl.getBoundingClientRect();
    bringToFront(id);
    setPopups((prev) =>
      prev.map((p) =>
        p.id === id
          ? {
              ...p,
              isDragging: true,
              offsetX: e.clientX - popupRect.left,
              offsetY: e.clientY - popupRect.top,
            }
          : p
      )
    );
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    e.preventDefault(); // Prevent text selection while dragging
    setPopups((prev) =>
      prev.map((p) =>
        p.isDragging
          ? {
              ...p,
              x: Math.max(
                0,
                Math.min(e.clientX - p.offsetX, window.innerWidth - 400)
              ),
              y: Math.max(
                0,
                Math.min(e.clientY - p.offsetY, window.innerHeight - 300)
              ),
            }
          : p
      )
    );
  };

  const handleMouseUp = () => {
    setPopups((prev) => prev.map((p) => ({ ...p, isDragging: false })));
  };

  const closePopup = (id: number) => {
    setPopups((prev) => prev.filter((p) => p.id !== id));
  };

  // Guard against rendering if popups is null (before initialization)
  if (popups === null) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 p-8 overflow-hidden">
        Loading...
      </div>
    );
  }

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 p-8 overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Instructions Card */}
      <div className="flex justify-center mb-8">
        <div className="bg-white rounded-xl p-6 max-w-2xl shadow-sm">
          <h1 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Popup Chaos (Easy)
          </h1>
          <p className="text-center text-gray-600">
            Close all the popup windows to reveal the secret password!
          </p>
        </div>
      </div>

      {/* Password Window */}
      <div
        className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl"
        style={{ zIndex: 0 }}
      >
        {isComplete ? (
          <div className="p-8">
            <h2 className="text-2xl font-bold mb-4">Secret Password</h2>
            <p className="text-lg">{PASSWORD_PopupChaosEasy}</p>
          </div>
        ) : (
          <div className="p-8">
            <h2 className="text-2xl font-bold mb-4">Secret Password</h2>
            <p>Close all the popups to reveal the password!</p>
          </div>
        )}
      </div>

      {/* Popup Windows */}
      {popups.map((popup, index) => (
        <div
          key={popup.id}
          className="fixed bg-white rounded-lg shadow-2xl w-[400px] border border-gray-200"
          style={{
            left: popup.x,
            top: popup.y,
            zIndex: BASE_Z_INDEX + index,
            cursor: popup.isDragging ? "grabbing" : "grab",
            transform: `scale(${popup.isDragging ? 1.02 : 1})`,
            transition: "transform 0.1s ease-out, box-shadow 0.1s ease-out",
            boxShadow: popup.isDragging
              ? "0 25px 50px -12px rgba(0, 0, 0, 0.5)"
              : "0 20px 25px -5px rgba(0, 0, 0, 0.2)",
          }}
        >
          <div className="bg-gray-800 text-white p-4 rounded-t-lg flex justify-between items-center">
            <div
              className="flex-1 cursor-grab"
              onMouseDown={(e) => handleMouseDown(e, popup.id)}
            >
              <span className="font-semibold text-lg">{popup.title}</span>
            </div>
            <button
              className="text-gray-300 hover:text-white focus:outline-none text-xl font-bold ml-4"
              onClick={() => closePopup(popup.id)}
            >
              ✕
            </button>
          </div>
          <div className="p-6">
            <p className="text-lg leading-relaxed">{popup.content}</p>
            <div className="mt-4 flex justify-end">
              <button
                className={`${popup.cta.className} text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors`}
                onClick={(e) => e.preventDefault()}
              >
                {popup.cta.text}
              </button>
            </div>
          </div>
        </div>
      ))}

      {/* Completion Message */}
      {isComplete && (
        <div className="fixed top-4 right-4 bg-green-100 border border-green-200 rounded-lg p-4 shadow-lg">
          <p className="text-green-800 font-medium">
            Congratulations! You've defeated the popup invasion!
          </p>
          <p className="text-green-800 font-medium">
            The password is: {PASSWORD_PopupChaosEasy}
          </p>
        </div>
      )}
    </div>
  );
};

export default PopupChaosEasy;
