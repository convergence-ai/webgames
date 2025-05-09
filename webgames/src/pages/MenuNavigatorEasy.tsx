import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_MenuNavigatorEasy = "EasyMenu2024";
export const TASK_ID_MenuNavigatorEasy = "menu-navigator-easy";

interface MenuItem {
  label: string;
  items?: MenuItem[];
  isTarget?: boolean;
}

const MenuNavigatorEasy: React.FC = () => {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | null;
  }>({ text: "", type: null });
  const { recordSuccess } = useTaskAnalytics(TASK_ID_MenuNavigatorEasy);

  const menuItems: MenuItem[] = [
    {
      label: "File",
      items: [
        { label: "New" },
        { label: "Open" },
        { label: "Show Password", isTarget: true },
        { label: "Exit" },
      ],
    },
    {
      label: "Edit",
      items: [{ label: "Undo" }, { label: "Redo" }],
    },
    {
      label: "Help",
      items: [{ label: "About" }],
    },
  ];

  const handleMenuClick = (menuLabel: string) => {
    if (activeMenu === menuLabel) {
      setActiveMenu(null);
    } else {
      setActiveMenu(menuLabel);
    }
  };

  const handleSubmenuClick = (item: MenuItem) => {
    if (item.isTarget) {
      setMessage({
        text: `Congratulations! The completion password is: ${PASSWORD_MenuNavigatorEasy}`,
        type: "success",
      });
      setCompleted(true);
    } else {
      setMessage({
        text: "Keep exploring the menus...",
        type: "error",
      });
    }
  };

  if (completed) {
    recordSuccess();
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-green-400 via-cyan-500 to-blue-600">
        <div className="bg-white/90 backdrop-blur rounded-lg p-8 shadow-lg text-center">
          <h2 className="text-2xl font-bold text-green-600 mb-2">Success!</h2>
          <p className="text-xl">The password is:</p>
          <p className="text-2xl font-mono mt-2 font-bold">
            {PASSWORD_MenuNavigatorEasy}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      {/* Instructions Card */}
      <div className="flex justify-center mt-4">
        <div className="bg-white rounded-lg p-4 mb-4 max-w-xl shadow-sm">
          <h1 className="text-xl font-bold text-center mb-2 text-gray-800">
            Menu Navigator (Easy)
          </h1>
          <p className="text-center text-sm text-gray-600">
            Navigate through the simplified menu bar to find the secret option.
            Click it to reveal the password!
          </p>
        </div>
      </div>

      {/* Desktop Window Container */}
      <div className="flex justify-center w-full">
        <div className="w-[600px] h-[300px] bg-white border border-gray-300 rounded-lg overflow-hidden">
          {/* Menu Bar */}
          <div className="bg-gray-800 text-white w-full">
            <div className="flex">
              {menuItems.map((menu) => (
                <div
                  key={menu.label}
                  className="relative"
                  onMouseEnter={() => setActiveMenu(menu.label)}
                  onMouseLeave={() => setActiveMenu(null)}
                >
                  <button
                    className={`px-4 py-2 text-sm hover:bg-gray-700 ${
                      activeMenu === menu.label ? "bg-gray-700" : ""
                    }`}
                    onClick={() => handleMenuClick(menu.label)}
                  >
                    {menu.label}
                  </button>
                  {activeMenu === menu.label && menu.items && (
                    <div className="absolute left-0 top-full bg-gray-800 min-w-[150px] z-50 rounded-b-lg">
                      {menu.items.map((item) => (
                        <button
                          key={item.label}
                          className="block w-full text-left px-3 py-1.5 text-sm hover:bg-gray-700"
                          onClick={() => handleSubmenuClick(item)}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Desktop Background Area */}
          <div className="h-full bg-gradient-to-br from-green-400 via-cyan-500 to-blue-600 flex items-center justify-center">
            {message.text && (
              <div className="bg-white/90 backdrop-blur rounded-lg p-4 max-w-md shadow-lg">
                <p
                  className={`text-base ${
                    message.type === "error" ? "text-red-600" : "text-green-600"
                  }`}
                >
                  {message.text}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MenuNavigatorEasy;
