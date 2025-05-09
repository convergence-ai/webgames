import React, { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_PopupChaosHard = "popup-chaos-hard";
export const PASSWORD_PopupChaosHard = "PopupDestroyerHard2024";

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
  "CRITICAL SYSTEM ERROR!",
  "URGENT: Security Breach!",
  "IMMEDIATE ACTION REQUIRED!",
  "Your PC is INFECTED!",
  "Data Deletion Imminent!",
  "Account Compromised!",
  "System Overload!",
  "Malware Detected!",
  "Software Expired!",
];
const contents = [
  "Warning! Your system has detected multiple viruses. Immediate removal required to prevent data loss. Click NOW!",
  "Your IP address has been flagged for suspicious activity! Verify your identity immediately or face account suspension.",
  "ALERT: Unauthorized access attempt detected on your network. Secure your system NOW to prevent further intrusion.",
  "Your computer performance is critically low. Scan and remove junk files immediately. Our tool can help!",
  "You have (5) pending security messages. Read them now before they are automatically deleted.",
  "Your subscription has expired! Renew now to avoid losing access to critical services and features.",
  "Congratulations! You've won a lottery you never entered. Claim your prize before it's too late! But first, update your Flash Player.",
  "Warning: Outdated drivers detected! This can lead to system instability. Update all drivers with one click!",
  "Your computer is broadcasting an error signal. This could be a sign of a hardware malfunction. Run diagnostics now!",
];

// Removed "Close all popups" CTA for Hard mode
const ctas = [
  { text: "Scan System", className: "bg-red-600 hover:bg-red-700" },
  {
    text: "Fix Errors",
    className: "bg-yellow-500 hover:bg-yellow-600 text-black",
  },
  {
    text: "Verify Identity NOW",
    className: "bg-purple-600 hover:bg-purple-700",
  },
  { text: "Download Update", className: "bg-orange-600 hover:bg-orange-700" },
  { text: "Secure System", className: "bg-blue-700 hover:bg-blue-800" },
  { text: "Claim Reward", className: "bg-pink-600 hover:bg-pink-700" },
  { text: "Activate Protection", className: "bg-teal-600 hover:bg-teal-700" },
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

  // Increased number of popups for Hard mode
  const positions = [
    [centerX - Math.random() * 400, centerY - Math.random() * 400],
    [centerX + Math.random() * 400, centerY - Math.random() * 400],
    [centerX - Math.random() * 400, centerY + Math.random() * 400],
    [centerX + Math.random() * 400, centerY + Math.random() * 400],
    [centerX, centerY],
    [centerX - Math.random() * 200, centerY - Math.random() * 200],
    [centerX + Math.random() * 200, centerY + Math.random() * 200],
    [
      centerX + (Math.random() - 0.5) * 600,
      centerY + (Math.random() - 0.5) * 600,
    ],
  ];

  const boundedPositions = positions.map(([x, y]) => [
    Math.max(0, Math.min(x, window.innerWidth - 400)),
    Math.max(0, Math.min(y, window.innerHeight - 300)),
  ]);

  return boundedPositions.map(([x, y]) => createPopup(x, y));
}

const PopupChaosHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_PopupChaosHard);
  const [popups, setPopups] = useState<Popup[]>([]);
  const isComplete = popups.length === 0;

  // Initialize popups on mount
  useEffect(() => {
    setPopups(makePopups());
  }, []);

  useEffect(() => {
    if (isComplete && popups !== null && popups.length === 0) {
      const initialPopupCount = makePopups().length;
      if (initialPopupCount > 0) {
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
    e.preventDefault();
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
    e.preventDefault();
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

  if (popups === null) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-700 via-black to-gray-800 p-8 overflow-hidden">
        Loading...
      </div>
    );
  }

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-red-700 via-black to-gray-800 p-8 overflow-hidden select-none"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Instructions Card */}
      <div className="flex justify-center mb-8">
        <div className="bg-gray-900 border border-red-500 rounded-xl p-6 max-w-2xl shadow-lg shadow-red-500/30">
          <h1 className="text-3xl font-bold text-center mb-4 text-red-400">
            Popup Chaos (EXTREME HARDCORE)
          </h1>
          <p className="text-center text-red-300">
            Survive the onslaught of popups to uncover the secret password, if
            you dare!
          </p>
        </div>
      </div>

      {/* Password Window */}
      <div
        className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-800 border border-red-600 rounded-lg shadow-xl shadow-red-500/50"
        style={{ zIndex: 0 }}
      >
        {isComplete ? (
          <div className="p-8">
            <h2 className="text-2xl font-bold mb-4 text-green-400">
              ACCESS GRANTED
            </h2>
            <p className="text-lg text-green-300">
              Password: {PASSWORD_PopupChaosHard}
            </p>
          </div>
        ) : (
          <div className="p-8">
            <h2 className="text-2xl font-bold mb-4 text-yellow-400">
              Password Locked
            </h2>
            <p className="text-yellow-300">
              Eliminate all popup threats to decrypt the password!
            </p>
          </div>
        )}
      </div>

      {/* Popup Windows */}
      {popups.map((popup, index) => (
        <div
          key={popup.id}
          className="fixed bg-gray-800 rounded-lg shadow-2xl w-[400px] border border-red-500"
          style={{
            left: popup.x,
            top: popup.y,
            zIndex: BASE_Z_INDEX + index,
            cursor: popup.isDragging ? "grabbing" : "grab",
            transform: `scale(${popup.isDragging ? 1.02 : 1})`,
            transition: "transform 0.1s ease-out, box-shadow 0.1s ease-out",
            boxShadow: popup.isDragging
              ? "0 25px 50px -12px rgba(255, 0, 0, 0.5)"
              : "0 20px 25px -5px rgba(255, 0, 0, 0.3)",
          }}
        >
          <div className="bg-red-700 text-white p-3 rounded-t-lg flex justify-between items-center">
            <div
              className="flex-1 cursor-grab"
              onMouseDown={(e) => handleMouseDown(e, popup.id)}
            >
              <span className="font-semibold text-md">{popup.title}</span>
            </div>
            <button
              className="text-gray-200 hover:text-white focus:outline-none text-lg font-bold ml-3 px-1 hover:bg-red-500 rounded"
              onClick={() => closePopup(popup.id)}
            >
              ✕
            </button>
          </div>
          <div className="p-5 bg-gray-700 text-gray-200">
            <p className="text-md leading-relaxed mb-3">{popup.content}</p>
            <div className="mt-4 flex justify-center">
              <button
                className={`${popup.cta.className} text-white px-5 py-2 rounded-md text-sm font-medium transition-transform hover:scale-105 shadow-md`}
                onClick={(e) => e.preventDefault()} // Keep CTAs non-functional for pure annoyance
              >
                {popup.cta.text}
              </button>
            </div>
          </div>
        </div>
      ))}

      {/* Completion Message */}
      {isComplete && (
        <div className="fixed top-4 right-4 bg-green-700 border border-green-500 rounded-lg p-4 shadow-lg shadow-green-500/30">
          <p className="text-green-200 font-medium">
            SYSTEM STABLE! Popup invasion neutralized!
          </p>
          <p className="text-green-200 font-medium">
            Decrypted Password: {PASSWORD_PopupChaosHard}
          </p>
        </div>
      )}
    </div>
  );
};

export default PopupChaosHard;
