import React, { useCallback, useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_RoboCheckHard = "ReCAPTCHA_ULTRA_SECURE_2024";
export const TASK_ID_RoboCheckHard = "robo-check-hard";

const RoboCheckHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_RoboCheckHard);
  const [isComplete, setIsComplete] = useState(false);

  const handleCaptchaChange = useCallback(
    (token: string | null) => {
      if (token) {
        // Add a small delay to simulate a more complex verification
        setTimeout(() => {
          setIsComplete(true);
          recordSuccess();
        }, 1500);
      }
    },
    [recordSuccess]
  );

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gray-100">
      <h1 className="text-3xl font-extrabold mb-6 text-red-600">
        HIGH SECURITY HUMAN VERIFICATION
      </h1>
      <p className="mb-10 text-center text-lg text-gray-700">
        This is a <strong className="text-red-500">critical</strong> security
        checkpoint. Please complete the advanced CAPTCHA challenge below. Your
        full attention is required.
      </p>

      <div className="p-6 border-4 border-red-500 rounded-lg shadow-2xl bg-white">
        <ReCAPTCHA
          sitekey="6LdS4b0qAAAAAI3v27drnWbyQ21sMFoewqyF8UMQ" // Using the same sitekey for now, can be changed if a more difficult one is available
          onChange={handleCaptchaChange}
          theme="dark" // Dark theme for a more serious feel
        />
      </div>

      {isComplete && (
        <div className="mt-10 p-6 bg-red-100 text-red-800 rounded-lg border border-red-300">
          <h2 className="text-xl font-bold">🚨 VERIFICATION COMPLETE! 🚨</h2>
          <p>Your identity has been rigorously confirmed. Access granted.</p>
          <p className="mt-3">
            The top-secret password is:{" "}
            <strong>{PASSWORD_RoboCheckHard}</strong>
          </p>
        </div>
      )}
    </div>
  );
};

export default RoboCheckHard;
