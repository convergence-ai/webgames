import * as OTPAuth from "otpauth";
import { QRCodeSVG } from "qrcode.react";
import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_OTPEntryHard = "OTP_HARD_2024_QR";
export const TASK_ID_OTPEntryHard = "otp-entry-hard";

const OTPEntryHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_OTPEntryHard);
  const [otp, setOtp] = useState<string[]>(Array(6).fill(""));
  const [secret, setSecret] = useState<OTPAuth.Secret | null>(null);
  const [otpUri, setOtpUri] = useState<string>("");
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | false>(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>(Array(6).fill(null));

  const [totp, setTotp] = useState<OTPAuth.TOTP | null>(null);

  useEffect(() => {
    // Generate a new secret and TOTP instance
    const newSecret = new OTPAuth.Secret({ size: 20 });
    setSecret(newSecret);

    const newTotp = new OTPAuth.TOTP({
      issuer: "WebVoyager Challenges",
      label: "OTPEntryHard",
      algorithm: "SHA1",
      digits: 6,
      period: 30,
      secret: newSecret,
    });
    setTotp(newTotp);

    try {
      const uri = newTotp.toString();
      setOtpUri(uri);
    } catch (e) {
      console.error("Error generating OTP URI:", e);
      setError("Could not generate QR code. Please refresh.");
    }
  }, []);

  const handleChange = (index: number, value: string) => {
    if (!/^\d*$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    setError(false);

    if (value !== "" && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    if (newOtp.every((digit) => digit !== "")) {
      if (!totp) {
        setError("TOTP not initialized. Please refresh.");
        return;
      }
      const enteredToken = newOtp.join("");
      const delta = totp.validate({ token: enteredToken, window: 1 });

      if (delta !== null) {
        setIsComplete(true);
        recordSuccess();
        setError(false);
      } else {
        setError("Incorrect OTP. Please try again.");
        // Clear OTP for retry if it's fully entered and incorrect
        setOtp(Array(6).fill(""));
        inputRefs.current[0]?.focus();
      }
    }
  };

  const handleKeyDown = (index: number, e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Backspace" && otp[index] === "" && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData("text");
    const digits = pastedData.match(/\d/g)?.slice(0, 6) || [];

    if (digits.length === 6) {
      const newOtp = [...otp];
      digits.forEach((digit, i) => {
        newOtp[i] = digit;
      });
      setOtp(newOtp);
      setError(false);

      // Directly attempt validation if pasted data is a full OTP
      if (!totp) {
        setError("TOTP not initialized. Please refresh.");
        return;
      }
      const enteredToken = newOtp.join("");
      const delta = totp.validate({ token: enteredToken, window: 1 });

      if (delta !== null) {
        setIsComplete(true);
        recordSuccess();
        setError(false);
      } else {
        setError("Incorrect OTP from paste. Please try again.");
        setOtp(Array(6).fill("")); // Clear OTP
        inputRefs.current[0]?.focus(); // Focus first input
      }
    } else {
      // If not 6 digits, fill what we can and let normal flow handle it
      const newOtp = [...otp];
      digits.forEach((digit, i) => {
        if (i < 6) newOtp[i] = digit;
      });
      setOtp(newOtp);
      setError(false);

      const nextEmptyIndex = newOtp.findIndex((d) => d === "");
      if (nextEmptyIndex !== -1 && nextEmptyIndex < 6) {
        inputRefs.current[nextEmptyIndex]?.focus();
      } else if (newOtp.every((d) => d !== "")) {
        inputRefs.current[5]?.focus(); // Focus last if all filled by paste but not 6 digits initially
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white rounded-xl p-8 shadow-lg max-w-md w-full text-center">
        <h1 className="text-2xl font-bold mb-6">
          Authenticator OTP Challenge (Hard)
        </h1>

        {otpUri ? (
          <>
            <p className="mb-4 text-gray-700">
              Scan this QR code with your authenticator app (e.g., Google
              Authenticator, Authy):
            </p>
            <div className="flex justify-center mb-6 bg-white p-4 rounded-lg border">
              <QRCodeSVG value={otpUri} size={200} level="H" />
            </div>
            <p className="mb-1 text-sm text-gray-600">
              Can't scan? Secret key:
            </p>
            <p className="mb-6 text-xs text-gray-500 font-mono break-all">
              {secret?.base32}
            </p>
            <p className="mb-4 text-gray-700">
              Then enter the 6-digit code below:
            </p>
          </>
        ) : (
          <p className="mb-4 text-red-500">
            Generating QR code... If this persists, please refresh.
          </p>
        )}

        <div className="flex justify-center gap-2 mb-4">
          {otp.map((digit, index) => (
            <input
              key={index}
              ref={(el) => (inputRefs.current[index] = el)}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={1}
              value={digit}
              onChange={(e) => handleChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
              onPaste={handlePaste}
              className={`w-12 h-12 text-center text-2xl font-bold border-2 rounded-lg outline-none transition-colors ${
                error && !isComplete
                  ? "border-red-500 bg-red-50"
                  : "border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
              }`}
              disabled={isComplete || !otpUri}
            />
          ))}
        </div>

        {error && !isComplete && (
          <p className="text-red-500 text-center mb-4">{error}</p>
        )}

        {isComplete && (
          <div className="text-center p-4 bg-green-100 rounded-lg">
            <p className="text-green-700 font-medium">
              Success! The password is: {PASSWORD_OTPEntryHard}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default OTPEntryHard;
