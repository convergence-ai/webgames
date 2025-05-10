import React, { useCallback, useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_RoboCheckEasy = "ReCAPTCHA_EASY_PEASY_2024";
export const TASK_ID_RoboCheckEasy = "robo-check-easy";

const RoboCheckEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_RoboCheckEasy);
  const [isComplete, setIsComplete] = useState(false);

  const handleCaptchaChange = useCallback(
    (token: string | null) => {
      if (token) {
        setIsComplete(true);
        recordSuccess();
      }
    },
    [recordSuccess]
  );

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-4">Prove You're Human (Easy)</h1>
      <p className="mb-8 text-center">
        Quickly complete the CAPTCHA challenge to show you're not a bot.
      </p>

      <ReCAPTCHA
        sitekey="6LdS4b0qAAAAAI3v27drnWbyQ21sMFoewqyF8UMQ" // Using the same sitekey for now, can be changed if a different one is needed for easy
        onChange={handleCaptchaChange}
      />

      {isComplete && (
        <div className="mt-8 p-4 bg-green-100 text-green-700 rounded-lg">
          <h2 className="text-lg font-bold">🎉 Well Done!</h2>
          <p>You've easily proven you're human.</p>
          <p className="mt-2">
            The secret password is: <strong>{PASSWORD_RoboCheckEasy}</strong>
          </p>
        </div>
      )}
    </div>
  );
};

export default RoboCheckEasy;
