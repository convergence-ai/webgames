import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

interface Item {
  name: string;
  emoji: string;
  location: "left" | "right" | "boat";
}

export const TASK_ID_WolfGoatCabbageHard = "wolf-goat-cabbage-hard";
export const PASSWORD_WolfGoatCabbageHard = "FoxesAreTricky";

const WolfGoatCabbageHard = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_WolfGoatCabbageHard);
  const [boatPosition, setBoatPosition] = useState<"left" | "right">("left");
  const [items, setItems] = useState<Item[]>([
    { name: "wolf", emoji: "🐺", location: "left" },
    { name: "goat", emoji: "🐐", location: "left" },
    { name: "cabbage", emoji: "🥬", location: "left" },
    { name: "fox", emoji: "🦊", location: "left" },
  ]);
  const [showPassword, setShowPassword] = useState(false);

  const getItemsInBoat = () => items.filter((item) => item.location === "boat");

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

    const checkDanger = (currentBankItems: Item[]) => {
      if (currentBankItems.length < 2) return false;

      const hasWolf = currentBankItems.some((item) => item.name === "wolf");
      const hasGoat = currentBankItems.some((item) => item.name === "goat");
      const hasCabbage = currentBankItems.some(
        (item) => item.name === "cabbage"
      );
      const hasFox = currentBankItems.some((item) => item.name === "fox");

      // Original dangers
      if (hasWolf && hasGoat) return true;
      if (hasGoat && hasCabbage) return true;
      // New dangers with Fox
      if (hasFox && hasWolf) return true; // Fox eats Wolf
      if (hasGoat && hasFox) return true; // Goat eats Fox

      return false;
    };

    if (boatPosition === "left") {
      return checkDanger(rightItems);
    } else {
      return checkDanger(leftItems);
    }
  };

  const loadItem = (itemName: string) => {
    if (getItemsInBoat().length >= 2) return; // Boat can carry up to 2 items

    setItems(
      items.map((item) => {
        if (item.name === itemName && item.location === boatPosition) {
          return { ...item, location: "boat" };
        }
        return item;
      })
    );
  };

  const unloadItem = (itemName: string) => {
    setItems((prevItems) => {
      const newItems = prevItems.map((item) => {
        if (item.name === itemName && item.location === "boat") {
          return { ...item, location: boatPosition };
        }
        return item;
      });

      if (newItems.every((i) => i.location === "right")) {
        setShowPassword(true);
        recordSuccess();
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
                  disabled={
                    boatPosition !== side ||
                    getItemsInBoat().length >= 2 ||
                    isGameOver
                  }
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
    const itemsInBoat = getItemsInBoat();
    return (
      <div
        className={`flex flex-col items-center transition-all duration-500 absolute ${
          boatPosition === "right" ? "right-36" : "left-36"
        }`}
      >
        <div className="min-h-[100px] flex flex-col items-center justify-center space-y-1">
          {itemsInBoat.length === 0 && (
            <span className="text-gray-500 italic">Boat empty</span>
          )}
          {itemsInBoat.map((item) => (
            <div key={item.name} className="flex items-center space-x-2 p-1">
              <span className="text-4xl">{item.emoji}</span>
              <button
                onClick={() => unloadItem(item.name)}
                className="px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-300 text-xs"
                disabled={isGameOver}
              >
                Unload
              </button>
            </div>
          ))}
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
              checkWinCondition();
            }}
            disabled={boatPosition === "left" || isGameOver}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            Move boat left
          </button>
          <button
            onClick={() => {
              setBoatPosition("right");
              checkWinCondition();
            }}
            disabled={boatPosition === "right" || isGameOver}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            Move boat right
          </button>
        </div>
      </div>
    );
  };

  const isGameOver = checkGameOver();

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">
          Wolf, Goat, Cabbage, and Fox (Hard)
        </h2>

        {showPassword ? (
          <div className="p-8 bg-green-50 border-2 border-green-200 rounded-lg">
            <h3 className="text-2xl font-bold text-green-800 mb-4">
              🎉 Congratulations!
            </h3>
            <p className="text-lg text-green-700 mb-2">
              You've masterfully transported everyone across the river!
            </p>
            <div className="mt-4 p-4 bg-green-100 rounded-lg">
              <p className="text-xl font-semibold text-green-800">
                Secret Password: {PASSWORD_WolfGoatCabbageHard}
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-4">
              <p className="text-gray-700">
                Help the farmer transport everything across the river! This time
                it's harder:
                <ul className="list-disc list-inside">
                  <li>The wolf will eat the goat if left alone.</li>
                  <li>The goat will eat the cabbage if left alone.</li>
                  <li>The fox will eat the wolf if left alone.</li>
                  <li>The goat will eat the fox if left alone.</li>
                  <li>The boat can carry up to two items at a time.</li>
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
                Game Over! The animals got into a tussle. Try again!
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default WolfGoatCabbageHard;
