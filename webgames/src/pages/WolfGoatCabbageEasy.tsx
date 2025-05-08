import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface Item {
  name: string;
  emoji: string;
  location: "left" | "right" | "boat";
}

export const TASK_ID_WolfGoatCabbageEasy = "wolf-goat-cabbage-easy";
export const PASSWORD_WolfGoatCabbageEasy = "GoatsAreFriends"; // Simplified password

const WolfGoatCabbageEasy = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_WolfGoatCabbageEasy);
  const [boatPosition, setBoatPosition] = useState<"left" | "right">("left");
  const [items, setItems] = useState<Item[]>([
    { name: "wolf", emoji: "🐺", location: "left" },
    { name: "goat", emoji: "🐐", location: "left" },
    // Cabbage removed for easy version
  ]);
  const [showPassword, setShowPassword] = useState(false);

  const getItemInBoat = () => items.find((item) => item.location === "boat");
  const hasItemInBoat = () => !!getItemInBoat();

  const checkWinCondition = () => {
    const allOnRight = items.every((item) => item.location === "right");
    if (allOnRight) {
      setShowPassword(true);
      recordSuccess();
    }
  };

  const checkGameOver = () => {
    const leftItems = items.filter((item) => item.location === "left");
    const rightItems = items.filter((item) => item.location === "right");

    // Only check for danger if the boat is on the opposite bank
    const checkDanger = (currentBankItems: Item[]) => {
      if (currentBankItems.length < 2) return false; // Need at least 2 items for danger

      const hasWolf = currentBankItems.some((item) => item.name === "wolf");
      const hasGoat = currentBankItems.some((item) => item.name === "goat");

      // Only wolf and goat conflict
      return hasWolf && hasGoat;
    };

    // Only check the bank that doesn't have the boat
    if (boatPosition === "left") {
      return checkDanger(rightItems);
    } else {
      return checkDanger(leftItems);
    }
  };

  const loadItem = (itemName: string) => {
    if (getItemInBoat()) return;

    setItems(
      items.map((item) => {
        if (item.name === itemName && item.location === boatPosition) {
          return { ...item, location: "boat" };
        }
        return item;
      })
    );
  };

  const unloadItem = () => {
    const itemInBoat = getItemInBoat();
    if (!itemInBoat) return;

    setItems((prevItems) => {
      const newItems = prevItems.map((item) => {
        if (item.name === itemInBoat.name) {
          return { ...item, location: boatPosition };
        }
        return item;
      });

      // Check win condition with the new state
      if (newItems.every((item) => item.location === "right")) {
        setShowPassword(true);
      }

      return newItems;
    });
  };

  const renderBank = (side: "left" | "right") => {
    return (
      <div className={`flex flex-col items-center space-y-2 w-32`}>
        <div className="min-h-[100px] flex flex-col items-center justify-center space-y-2">
          {items
            .filter((item) => item.location === side)
            .map((item) => (
              <div key={item.name} className="flex items-center space-x-2">
                <span className="text-4xl">{item.emoji}</span>
                <button
                  onClick={() => loadItem(item.name)}
                  className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:bg-gray-300 text-sm"
                  disabled={boatPosition !== side || hasItemInBoat()}
                >
                  Load {item.name}
                </button>
              </div>
            ))}
        </div>
        <div className="h-4 w-full bg-green-600"></div>
      </div>
    );
  };

  const renderBoat = () => {
    const itemInBoat = getItemInBoat();
    return (
      <div
        className={`flex flex-col items-center transition-all duration-500 absolute ${
          boatPosition === "right" ? "right-36" : "left-36"
        }`}
      >
        <div className="min-h-[60px] flex items-center justify-center">
          {itemInBoat && <span className="text-4xl">{itemInBoat.emoji}</span>}
        </div>
        <div className="text-4xl">⛵</div>
      </div>
    );
  };

  const renderControls = () => {
    return (
      <div className="flex flex-col items-center space-y-4">
        <div className="flex space-x-4">
          <button
            onClick={() => {
              setBoatPosition("left");
              checkWinCondition(); // Check win after moving boat
            }}
            disabled={boatPosition === "left" || isGameOver}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            Move boat left
          </button>
          <button
            onClick={() => {
              setBoatPosition("right");
              checkWinCondition(); // Check win after moving boat
            }}
            disabled={boatPosition === "right" || isGameOver}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            Move boat right
          </button>
        </div>
        <button
          onClick={unloadItem}
          disabled={!hasItemInBoat() || isGameOver}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300"
        >
          Unload
        </button>
      </div>
    );
  };

  const isGameOver = checkGameOver();

  // Call checkWinCondition whenever items or boatPosition changes to catch win state
  // This is implicitly handled by calling it after boat moves and after item unloads.

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">Wolf and Goat (Easy)</h2>

        {showPassword ? (
          <div className="p-8 bg-green-50 border-2 border-green-200 rounded-lg">
            <h3 className="text-2xl font-bold text-green-800 mb-4">
              🎉 Congratulations!
            </h3>
            <p className="text-lg text-green-700 mb-2">
              You've successfully transported everyone across the river!
            </p>
            <div className="mt-4 p-4 bg-green-100 rounded-lg">
              <p className="text-xl font-semibold text-green-800">
                Secret Password: {PASSWORD_WolfGoatCabbageEasy}
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-4">
              <p className="text-gray-700">
                Help the farmer transport the wolf and goat across the river!
                But be careful:
                <ul className="list-disc list-inside">
                  <li>The wolf will eat the goat if left alone</li>
                  <li>The boat can only carry one item at a time</li>
                </ul>
              </p>
            </div>

            <div className="relative mb-8">
              <div className="absolute inset-0 bg-blue-200 mt-20 h-20"></div>
              <div className="relative flex justify-between items-center">
                {renderBank("left")}
                {renderBoat()}
                {renderBank("right")}
              </div>
            </div>

            {renderControls()}

            {isGameOver && (
              <div className="mt-4 p-4 bg-red-100 border border-red-200 rounded text-red-700">
                Game Over! The wolf got hungry. Try again!
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default WolfGoatCabbageEasy;
