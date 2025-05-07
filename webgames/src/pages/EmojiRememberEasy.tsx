import { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

const EMOJI_SEQUENCE_EASY = ["🌟", "🎨", "🌈", "🎭", "🎪"]; // Shorter sequence
const DECOY_EMOJIS_EASY = [
  "🎯",
  "🎳",
  "🎰",
  "🎱",
  "🎸",
  "🎺",
  "🎻",
  "🎹",
  "🥁",
  "🎷",
]; // Fewer decoys for simplicity

export const TASK_ID_EmojiRememberEasy = "emoji-remember-easy";
export const PASSWORD_EmojiRememberEasy = "MemoryEasyPeasy";

const EmojiRememberEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_EmojiRememberEasy);
  const [currentStep, setCurrentStep] = useState(0);
  const [displayedEmojis, setDisplayedEmojis] = useState<string[]>([]);
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | null;
  }>({ text: "", type: null });
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    const generateDisplayEmojis = (correctEmoji: string) => {
      // Ensure decoys don't include emojis already shown or the current correct one
      const emojisToExclude = [
        ...EMOJI_SEQUENCE_EASY.slice(0, currentStep),
        correctEmoji,
      ];
      const availableDecoys = DECOY_EMOJIS_EASY.filter(
        (emoji) => !emojisToExclude.includes(emoji)
      );

      const otherEmojis = availableDecoys
        .sort(() => Math.random() - 0.5)
        .slice(0, 2); // Fewer decoy options

      return [...otherEmojis, correctEmoji].sort(() => Math.random() - 0.5);
    };

    if (!isComplete) {
      setDisplayedEmojis(
        generateDisplayEmojis(EMOJI_SEQUENCE_EASY[currentStep])
      );
    }
  }, [currentStep, isComplete]);

  const handleEmojiClick = (emoji: string) => {
    if (emoji === EMOJI_SEQUENCE_EASY[currentStep]) {
      if (currentStep === EMOJI_SEQUENCE_EASY.length - 1) {
        setIsComplete(true);
        setMessage({
          text: `Congratulations! You have completed the sequence! 🎉 The secret password is: ${PASSWORD_EmojiRememberEasy}`,
          type: "success",
        });
      } else {
        setCurrentStep((prev) => prev + 1);
        setMessage({ text: "", type: null });
      }
    } else {
      setMessage({ text: "Wrong emoji! Starting over...", type: "error" });
      setCurrentStep(0);
    }
  };

  useEffect(() => {
    if (isComplete) {
      recordSuccess();
    }
  }, [isComplete, recordSuccess]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-green-100 to-lime-100 p-4">
      <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-3xl font-bold text-center mb-6 text-green-600">
          Emoji Remember (Easy)
        </h1>

        <div className="text-center mb-6">
          <p className="text-lg mb-2">
            Progress: {currentStep + 1}/{EMOJI_SEQUENCE_EASY.length}
          </p>
          <p className="text-sm text-gray-600">
            {/* Display previously correctly selected emojis */}
            {currentStep > 0
              ? EMOJI_SEQUENCE_EASY.slice(0, currentStep).join(" ")
              : "Click the first emoji!"}
          </p>
        </div>

        {message.text && (
          <div
            className={`mb-4 p-3 rounded-lg text-center ${
              message.type === "error"
                ? "bg-red-100 text-red-700"
                : message.type === "success"
                ? "bg-green-100 text-green-700"
                : ""
            }`}
          >
            {message.text}
          </div>
        )}

        {!isComplete && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            {displayedEmojis.map((emoji, index) => (
              <button
                key={index}
                onClick={() => handleEmojiClick(emoji)}
                className="text-4xl p-4 bg-green-50 hover:bg-green-100 rounded-lg transition-colors duration-200"
                aria-label={`Emoji ${emoji}`}
              >
                {emoji}
              </button>
            ))}
          </div>
        )}

        {!isComplete && (
          <p className="text-sm text-center text-gray-600">
            Click the correct emoji to progress. Choose wrong and you'll start
            over!
          </p>
        )}
      </div>
    </div>
  );
};

export default EmojiRememberEasy;
