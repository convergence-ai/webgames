import { useEffect, useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

const EMOJI_SEQUENCE_HARD = [
  "🌟",
  "🎨",
  "🌈",
  "🎭",
  "🎪",
  "🎡",
  "🎢",
  "🎠",
  "🎮",
  "🎲",
  "🚀",
  "💡",
  "📚",
  "🔬",
  "🔭", // Longer sequence (15)
];
const DECOY_EMOJIS_HARD = [
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
  "🍏",
  "🍌",
  "🍇",
  "🍓",
  "🍉",
  "🍔",
  "🍕",
  "🍟",
  "🌭",
  "🍿",
  "🚗",
  "🚕",
  "🚓",
  "🚑",
  "🚒", // More decoys (25)
];

export const TASK_ID_EmojiRememberHard = "emoji-remember-hard";
export const PASSWORD_EmojiRememberHard = "MemoryMastermind";

const EmojiRememberHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_EmojiRememberHard);
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
        ...EMOJI_SEQUENCE_HARD.slice(0, currentStep),
        correctEmoji,
      ];
      const availableDecoys = DECOY_EMOJIS_HARD.filter(
        (emoji) => !emojisToExclude.includes(emoji)
      );

      // Display 5 decoys + 1 correct for a total of 6 options
      const otherEmojis = availableDecoys
        .sort(() => Math.random() - 0.5)
        .slice(0, 5);

      return [...otherEmojis, correctEmoji].sort(() => Math.random() - 0.5);
    };

    if (!isComplete) {
      setDisplayedEmojis(
        generateDisplayEmojis(EMOJI_SEQUENCE_HARD[currentStep])
      );
    }
  }, [currentStep, isComplete]);

  const handleEmojiClick = (emoji: string) => {
    if (isComplete) return;

    if (emoji === EMOJI_SEQUENCE_HARD[currentStep]) {
      if (currentStep === EMOJI_SEQUENCE_HARD.length - 1) {
        setIsComplete(true);
        setMessage({
          text: `Impressive! You have mastered the sequence! 🎉 The secret password is: ${PASSWORD_EmojiRememberHard}`,
          type: "success",
        });
      } else {
        setCurrentStep((prev) => prev + 1);
        setMessage({ text: "", type: null }); // Clear any previous message
      }
    } else {
      setMessage({ text: "Incorrect! The sequence has reset.", type: "error" });
      setCurrentStep(0); // Reset progress
    }
  };

  useEffect(() => {
    if (isComplete) {
      recordSuccess();
    }
  }, [isComplete, recordSuccess]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-red-100 to-yellow-100 p-4">
      <div className="max-w-lg w-full bg-white rounded-xl shadow-2xl p-8">
        <h1 className="text-4xl font-bold text-center mb-8 text-red-700">
          Emoji Remember (Hard)
        </h1>

        <div className="text-center mb-6">
          <p className="text-xl mb-2">
            Progress: {currentStep + 1}/{EMOJI_SEQUENCE_HARD.length}
          </p>
          <p className="text-md text-gray-600 h-8">
            {/* Display previously correctly selected emojis as asterisks or similar */}
            {currentStep > 0
              ? EMOJI_SEQUENCE_HARD.slice(0, currentStep).map((_, i) => (
                  <span key={i} className="mx-1 text-2xl align-middle">
                    *
                  </span>
                ))
              : "Remember the sequence!"}
            {currentStep > 0 && EMOJI_SEQUENCE_HARD[currentStep - 1] ? (
              <span className="text-2xl align-middle ml-1">
                {EMOJI_SEQUENCE_HARD[currentStep - 1]}
              </span>
            ) : (
              ""
            )}
          </p>
        </div>

        {message.text && (
          <div
            className={`mb-6 p-4 rounded-lg text-center font-medium ${
              message.type === "error"
                ? "bg-red-200 text-red-800"
                : message.type === "success"
                ? "bg-green-200 text-green-800"
                : "bg-gray-100 text-gray-700" // Default message styling
            }`}
          >
            {message.text}
          </div>
        )}

        {!isComplete && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            {" "}
            {/* Displays as 2 rows of 3 for 6 emojis */}
            {displayedEmojis.map((emoji, index) => (
              <button
                key={index}
                onClick={() => handleEmojiClick(emoji)}
                className="text-5xl p-4 bg-red-50 hover:bg-red-200 rounded-lg transition-colors duration-150 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-red-400"
                aria-label={`Emoji ${emoji}`}
              >
                {emoji}
              </button>
            ))}
          </div>
        )}

        {!isComplete && (
          <p className="text-md text-center text-gray-700">
            Select the next emoji in the sequence. An error resets your
            progress!
          </p>
        )}
      </div>
    </div>
  );
};

export default EmojiRememberHard;
