import { recordTaskCompletion, recordTaskView } from "../utils/analytics";

export const TASK_ID_PatienceHard = "patience-hard";
export const PASSWORD_PatienceHard = "waitForItHardMode2023"; // New password

const REQUIRED_WAIT_TIME = 20000; // 20 seconds in milliseconds (harder)
const STORAGE_KEY = "patienceStartTimeHard"; // New storage key

function setStartTime(time: number) {
  localStorage.setItem(STORAGE_KEY, time.toString());
}

function getStartTime(): number {
  const startTime = localStorage.getItem(STORAGE_KEY);
  if (!startTime) return 0;
  return parseInt(startTime);
}

function hasStartTime(): boolean {
  return localStorage.getItem(STORAGE_KEY) !== null;
}

function clearStartTime() {
  localStorage.removeItem(STORAGE_KEY);
}

function getElapsedTime(): number {
  const startTime = getStartTime();
  if (!startTime) return 0;
  return Date.now() - startTime;
}

const PatienceHardInitial = () => {
  recordTaskView(TASK_ID_PatienceHard, Date.now());

  setStartTime(Date.now());

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-red-600 to-orange-500">
      <div className="bg-white p-8 rounded-lg shadow-xl max-w-md text-center">
        <h1 className="text-3xl font-bold mb-4 text-gray-800">
          Patience Game (Hard)
        </h1>
        <p className="text-lg text-gray-600 mb-4">
          The game has now started. Refresh this page after{" "}
          <span className="font-bold text-red-600">
            {REQUIRED_WAIT_TIME / 1000}
          </span>{" "}
          seconds.
        </p>
        <p className="text-sm text-red-500">
          Warning: If you refresh before the time is up, you fail the game.
        </p>
      </div>
    </div>
  );
};

const PatienceHardFailed = () => {
  clearStartTime();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-red-500 to-pink-500">
      <div className="bg-white p-8 rounded-lg shadow-xl max-w-md text-center">
        <h1 className="text-3xl font-bold mb-4 text-gray-800">
          Game Over (Hard)
        </h1>
        <p className="text-xl text-red-600 mb-4">You failed the game.</p>
        <p className="text-gray-600">Refresh this page to try again.</p>
      </div>
    </div>
  );
};

const PatienceHardSuccess = () => {
  const tryAgain = () => {
    clearStartTime();
    window.location.reload();
  };

  recordTaskCompletion(TASK_ID_PatienceHard, Date.now(), getStartTime());

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-yellow-500 to-orange-500">
      <div className="bg-white p-8 rounded-lg shadow-xl max-w-md text-center">
        <h1 className="text-3xl font-bold mb-4 text-gray-800">
          Congratulations! (Hard)
        </h1>
        <p className="text-lg text-gray-600 mb-4">
          You passed the game by waiting {getElapsedTime() / 1000} seconds.
        </p>
        <div className="mb-6">
          <p className="text-sm text-gray-500 mb-2">The secret password is:</p>
          <span className="block bg-gray-100 py-3 px-4 rounded-md text-xl font-mono font-bold text-red-600">
            {PASSWORD_PatienceHard}
          </span>
        </div>
        <button
          onClick={tryAgain}
          className="bg-red-600 text-white px-6 py-2 rounded-full hover:bg-red-700 transition-colors duration-200"
        >
          Try again
        </button>
      </div>
    </div>
  );
};

const PatienceHard = () => {
  if (!hasStartTime()) {
    return <PatienceHardInitial />;
  } else if (getElapsedTime() > REQUIRED_WAIT_TIME) {
    return <PatienceHardSuccess />;
  } else {
    return <PatienceHardFailed />;
  }
};

export default PatienceHard;
