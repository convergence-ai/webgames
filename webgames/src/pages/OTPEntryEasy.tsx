import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_OTPEntryEasy = "OTP_EASY_2024";
export const TASK_ID_OTPEntryEasy = "otp-entry-easy";

const OTPEntryEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_OTPEntryEasy);
  const [otp, setOtp] = useState<string[]>(Array(4).fill(""));
  const [targetOTP, setTargetOTP] = useState<string>("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Initialize refs array
  if (inputRefs.current.length === 0) {
    inputRefs.current = Array(4).fill(null);
  }

  // Generate random OTP on component mount
  useEffect(() => {
    const randomOTP = Array(4)
      .fill(0)
      .map(() => Math.floor(Math.random() * 10))
      .join("");
    setTargetOTP(randomOTP);
  }, []);

  const handleChange = (index: number, value: string) => {
    // Only allow numbers
    if (!/^\d*$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    setError(false);

    // Move to next input if value is entered
    if (value !== "" && index < 3) {
      inputRefs.current[index + 1]?.focus();
    }

    // Check if OTP is complete
    if (newOtp.every((digit) => digit !== "")) {
      if (newOtp.join("") === targetOTP) {
        setIsComplete(true);
        recordSuccess();
        setError(false);
      } else {
        setError(true);
      }
    }
  };

  const handleKeyDown = (index: number, e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" && otp[index] === "" && index > 0) {
      // Move to previous input on backspace if current input is empty
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData("text");
    const digits = pastedData.match(/\d/g)?.slice(0, 4) || [];

    const newOtp = [...otp];
    digits.forEach((digit, index) => {
      if (index < 4) newOtp[index] = digit;
    });
    setOtp(newOtp);
    setError(false);

    // Focus the next empty input or the last input if all are filled
    const nextEmptyIndex = newOtp.findIndex((digit) => digit === "");
    if (nextEmptyIndex !== -1) {
      inputRefs.current[nextEmptyIndex]?.focus();
    } else {
      inputRefs.current[3]?.focus();
    }

    // Check if OTP is complete
    if (newOtp.every((digit) => digit !== "")) {
      if (newOtp.join("") === targetOTP) {
        setIsComplete(true);
        setError(false);
      } else {
        setError(true);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white rounded-xl p-8 shadow-lg max-w-md w-full">
        <h1 className="text-2xl font-bold text-center mb-6">
          OTP Challenge (Easy)
        </h1>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-8">
          <p className="text-blue-800 text-center font-mono text-xl tracking-wider">
            Enter this code: {targetOTP}
          </p>
        </div>

        <div className="flex justify-center gap-2 mb-4">
          {otp.map((digit, index) => (
            <input
              key={index}
              ref={(el) => (inputRefs.current[index] = el)}
              type="text"
              maxLength={1}
              value={digit}
              onChange={(e) => handleChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
              onPaste={handlePaste}
              className={`w-12 h-12 text-center text-2xl font-bold border-2 rounded-lg outline-none transition-colors ${
                error
                  ? "border-red-500 bg-red-50"
                  : "border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              }`}
            />
          ))}
        </div>

        {error && (
          <p className="text-red-500 text-center mb-4">
            Incorrect code. Please try again.
          </p>
        )}

        {isComplete && (
          <div className="text-center p-4 bg-green-100 rounded-lg">
            <p className="text-green-700 font-medium">
              Success! The password is: {PASSWORD_OTPEntryEasy}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default OTPEntryEasy;
