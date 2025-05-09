import React, { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
export const PASSWORD_IframeNestEasy = "NestedVoyagerEasy";
export const TASK_ID_IframeNestEasy = "iframe-nest-easy";

const IframeNestEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_IframeNestEasy);
  const [isComplete, setIsComplete] = useState(false);
  const iframeStyle = {
    width: "90%",
    height: "90%",
    border: "2px solid #333",
    borderRadius: "8px",
    margin: "20px auto",
    display: "block",
  };

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === "CHALLENGE_COMPLETE") {
        setIsComplete(true);
        recordSuccess();
      }
    };

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [recordSuccess]);

  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        padding: "20px",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(255, 255, 255, 0.9)",
          padding: "15px",
          borderRadius: "8px",
          fontSize: "16px",
          zIndex: 1000,
        }}
      >
        {!isComplete ? (
          <div>
            Find and click the button hidden in the nested iframes! (Easy
            version - fewer frames)
          </div>
        ) : (
          <span style={{ color: "#4CAF50", fontWeight: "bold" }}>
            Challenge Complete! 🎉 Password: {PASSWORD_IframeNestEasy}
          </span>
        )}
      </div>
      <iframe
        src={`${window.location.origin}/iframe-content/2`}
        style={iframeStyle}
        title="Nested Iframe Easy"
      />
    </div>
  );
};

export default IframeNestEasy;
