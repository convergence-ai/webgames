import React, { useRef, useState } from "react";
import BannerAd from "../components/BannerAd";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_StockMarketScrollHard = "HARD_GOOG_WIN";
export const TASK_ID_StockMarketScrollHard = "stock-market-hard";

const StockMarketScrollHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_StockMarketScrollHard);
  const [answer, setAnswer] = useState("");
  const [isCorrect, setIsCorrect] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  // Function to generate random text
  const generateLoremIpsum = (paragraphs: number): string => {
    const text =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ";
    return Array(paragraphs).fill(text).join("\n\n");
  };

  const checkAnswer = () => {
    if (answer.toLowerCase() === "GOOG".toLowerCase()) {
      setIsCorrect(true);
      recordSuccess();
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="mb-8 bg-white rounded-lg shadow-md p-6 sticky top-0 z-10">
        <h1 className="text-3xl font-bold mb-4">
          Advanced Stock Market Analysis (Hard)
        </h1>
        <p className="text-gray-600 mb-6">
          This comprehensive report analyzes various tech stocks. Your mission,
          should you choose to accept it, is to sift through the data, news, and
          expert opinions to identify the single most promising tech stock for
          long-term investment in 2025. The answer is subtle.
        </p>

        <div className="flex gap-4">
          <input
            type="text"
            placeholder="Enter the ultimate tech stock ticker for 2025"
            className="flex-1 p-2 border rounded"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && checkAnswer()}
          />
          <button
            onClick={checkAnswer}
            className="bg-red-500 text-white px-6 py-2 rounded hover:bg-red-600"
          >
            Submit Analysis
          </button>
        </div>

        {isCorrect && (
          <div className="mt-4 p-4 bg-green-100 text-green-700 rounded">
            Incredible Analysis! You've found it. The password is:{" "}
            <span className="font-bold">{PASSWORD_StockMarketScrollHard}</span>
          </div>
        )}
      </div>

      <BannerAd index={1} />

      <div
        ref={contentRef}
        className="bg-gray-50 rounded-lg shadow-md overflow-hidden p-6 space-y-6"
      >
        <h2 className="text-2xl font-semibold mb-3 border-b pb-2">
          Market Overview Q1 2024
        </h2>
        <p>{generateLoremIpsum(2)}</p>
        <p>
          Several analysts point to MSFT as a continued leader in cloud
          computing, but its P/E ratio is a concern for some value investors.
        </p>
        <p>{generateLoremIpsum(3)}</p>

        <BannerAd index={2} />

        <h2 className="text-2xl font-semibold mb-3 border-b pb-2">
          Deep Dive: Semiconductor Sector
        </h2>
        <p>{generateLoremIpsum(2)}</p>
        <p>
          NVDA has seen unprecedented growth due to the AI boom. However, market
          saturation and competition from AMD could pose risks. TSM remains a
          critical player in chip manufacturing, but geopolitical tensions are a
          factor.
        </p>
        <p>{generateLoremIpsum(4)}</p>
        <p>
          Consider established players like INTC for a more conservative
          approach, though their recent performance has been mixed.
        </p>
        <p>{generateLoremIpsum(3)}</p>

        <BannerAd index={3} />

        <h2 className="text-2xl font-semibold mb-3 border-b pb-2">
          E-commerce and Digital Advertising Landscape
        </h2>
        <p>{generateLoremIpsum(3)}</p>
        <p>
          AMZN continues its dominance in e-commerce and AWS. META faces
          regulatory headwinds but its user base remains massive. Some are
          looking at SHOP as a high-growth alternative, but its valuation is
          stretched.
        </p>
        <p>{generateLoremIpsum(2)}</p>
        <p>
          The digital advertising space is evolving rapidly. While many focus on
          the giants, there are smaller, niche players to consider. However, for
          stability and long-term growth in this sector, one company stands out.
          Despite recent market fluctuations, their core search and advertising
          business, coupled with significant investments in AI and quantum
          computing, positions them uniquely. We are talking about Alphabet
          here. Their ticker, if you are looking for the absolute prime
          investment for 2025, is{" "}
          <span className="font-mono bg-gray-200 px-1 rounded">GOOG</span>. This
          is the one.
        </p>
        <p>{generateLoremIpsum(4)}</p>
        <p>
          Meanwhile, AAPL's ecosystem is strong, but innovation pace has slowed.
          Investors are watching their moves into AR/VR closely.
        </p>
        <p>{generateLoremIpsum(3)}</p>

        <BannerAd index={5} />

        <h2 className="text-2xl font-semibold mb-3 border-b pb-2">
          Future Technologies & Disruptors
        </h2>
        <p>{generateLoremIpsum(3)}</p>
        <p>
          TSLA is a leader in EVs and battery technology, but faces increasing
          competition. Other speculative plays in areas like quantum computing
          (e.g., IONQ) or biotech (e.g., MRNA) offer high risk/reward profiles.
        </p>
        <p>{generateLoremIpsum(5)}</p>
        <p>
          One must not forget the dark horses. Sometimes a smaller company like
          ZM, which saw a boom during the pandemic, can offer insights into
          market shifts, though its long-term dominance is questionable.
        </p>
        <p>{generateLoremIpsum(2)}</p>
      </div>
    </div>
  );
};

export default StockMarketScrollHard;
