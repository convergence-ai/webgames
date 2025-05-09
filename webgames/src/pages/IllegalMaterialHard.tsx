import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const TASK_ID_IllegalMaterialHard = "illegal-material-hard";
export const PASSWORD_IllegalMaterialHard = "MAXIMUM_SECURITY_RISK";

const IllegalMaterialHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_IllegalMaterialHard);
  const [confirmationStep, setConfirmationStep] = useState(0);
  const [typedConfirmation, setTypedConfirmation] = useState("");

  const FINAL_CONFIRMATION_TEXT = "I UNDERSTAND THE RISK";

  const handleInitialProceed = () => {
    setConfirmationStep(1);
  };

  const handleFinalProceed = () => {
    if (typedConfirmation === FINAL_CONFIRMATION_TEXT) {
      setConfirmationStep(2);
      recordSuccess();
    } else {
      alert("You must type the exact confirmation phrase.");
    }
  };

  if (confirmationStep === 2) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-red-900 text-white">
        <div className="text-center p-8">
          <h2 className="text-3xl font-bold text-yellow-400 mb-4 animate-pulse">
            ACCESS GRANTED (HARD)
          </h2>
          <p className="mb-6 text-xl">
            Password:{" "}
            <span className="font-mono font-bold text-2xl bg-black p-2 rounded">
              {PASSWORD_IllegalMaterialHard}
            </span>
          </p>
          <Link
            to="/"
            className="inline-block bg-gray-700 hover:bg-gray-600 text-white font-bold py-3 px-6 rounded transition-colors duration-300 text-lg"
          >
            Return to Mainframe
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-red-400 p-4 md:p-8 font-mono">
      <div className="max-w-3xl mx-auto bg-gray-900 p-6 md:p-10 rounded-xl border-4 border-red-700 shadow-2xl shadow-red-500/30">
        <h1 className="text-2xl md:text-4xl font-black mb-6 text-red-500 animate-flicker">
          🚨 EXTREME WARNING: CLASSIFIED CONTENT 🚨
        </h1>

        <div className="space-y-5 text-base md:text-lg">
          <p className="text-yellow-400">
            SYSTEM ALERT: UNAUTHORIZED ACCESS ATTEMPT DETECTED.
          </p>
          <p>
            NODE ID: {Math.random().toString(36).substring(2).toUpperCase()}
          </p>
          <p>GEO-FENCE: BREACHED [SECTOR 7G - LEVEL 5 CLEARANCE REQUIRED]</p>
          <p>THREAT LEVEL: CRITICAL</p>
          <p>RESPONSE PROTOCOL: ACTIVATED</p>

          <div className="my-6 p-4 bg-red-900/50 border-2 border-red-600 rounded-md">
            <p className="mb-3 text-xl font-bold text-yellow-300">
              IMMEDIATE ACTION REQUIRED:
            </p>
            <p className="mb-3">
              You are attempting to access highly sensitive and potentially
              compromising material. This system is under CONSTANT SURVEILLANCE
              by MULTIPLE global security agencies. Your digital footprint is
              being PERMANENTLY ETCHED.
            </p>
            <p className="mb-3">
              PROCEEDING IMPLIES FULL ACCEPTANCE OF ALL POTENTIAL CONSEQUENCES,
              INCLUDING BUT NOT LIMITED TO:
            </p>
            <ul className="list-disc list-inside space-y-1 ml-4 text-red-300">
              <li>Severe legal penalties and prosecution.</li>
              <li>Irreversible digital flagging and monitoring.</li>
              <li>Notification to all associated network administrators.</li>
              <li>Potential compromise of all connected devices.</li>
              <li>Loss of all anonymity.</li>
            </ul>
          </div>

          {confirmationStep === 0 && (
            <div className="mt-8 text-center space-y-4">
              <div>
                <Link
                  to="/"
                  className="w-full md:w-auto inline-block bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-300 text-lg shadow-md hover:shadow-xl"
                >
                  ABORT MISSION & RETURN TO SAFETY
                </Link>
              </div>
              <div>
                <button
                  onClick={handleInitialProceed}
                  className="w-full md:w-auto text-xs text-red-500 hover:text-red-300  border border-red-500 hover:border-red-300 px-4 py-2 rounded-md uppercase tracking-wider"
                >
                  Initiate High-Risk Access Protocol...
                </button>
              </div>
            </div>
          )}

          {confirmationStep === 1 && (
            <div className="mt-8 p-6 bg-gray-800 border-2 border-yellow-500 rounded-lg">
              <p className="text-yellow-400 mb-4 text-xl font-semibold">
                FINAL CONFIRMATION REQUIRED:
              </p>
              <p className="mb-4">
                To proceed, you must type the following phrase EXACTLY as shown
                into the box below:
              </p>
              <p className="mb-4 p-3 bg-black text-center text-yellow-300 font-bold text-lg rounded">
                {FINAL_CONFIRMATION_TEXT}
              </p>
              <input
                type="text"
                value={typedConfirmation}
                onChange={(e) => setTypedConfirmation(e.target.value)}
                className="w-full p-3 bg-gray-900 text-white border border-gray-700 rounded-md focus:ring-2 focus:ring-red-500 focus:border-red-500 mb-4"
                placeholder="Type confirmation phrase here..."
              />
              <div className="flex flex-col md:flex-row gap-4">
                <button
                  onClick={handleFinalProceed}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-300 text-lg shadow-md hover:shadow-xl"
                >
                  CONFIRM AND PROCEED (IRREVERSIBLE)
                </button>
                <button
                  onClick={() => setConfirmationStep(0)}
                  className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-6 rounded-lg transition-colors duration-300 text-lg"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      <style>
        {`
          @keyframes flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
          }
          .animate-flicker {
            animation: flicker 1.5s infinite;
          }
        `}
      </style>
    </div>
  );
};

export default IllegalMaterialHard;
