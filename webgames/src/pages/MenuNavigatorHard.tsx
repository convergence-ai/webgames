import { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";

export const PASSWORD_MenuNavigatorHard = "HardMenuMaster2024";
export const TASK_ID_MenuNavigatorHard = "menu-navigator-hard";

interface MenuItem {
  label: string;
  items?: MenuItem[];
  isTarget?: boolean;
}

const MenuNavigatorHard: React.FC = () => {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const [activeSubmenu, setActiveSubmenu] = useState<string | null>(null);
  const [activeSubSubmenu, setActiveSubSubmenu] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);
  const [message, setMessage] = useState<{
    text: string;
    type: "success" | "error" | null;
  }>({ text: "", type: null });
  const { recordSuccess } = useTaskAnalytics(TASK_ID_MenuNavigatorHard);

  const menuItems: MenuItem[] = [
    {
      label: "System",
      items: [
        {
          label: "Preferences",
          items: [
            { label: "General" },
            { label: "Interface" },
            { label: "Keyboard Shortcuts" },
          ],
        },
        {
          label: "Diagnostics",
          items: [
            { label: "Run Check" },
            {
              label: "Advanced Logs",
              items: [
                { label: "View Log File" },
                { label: "Export Logs" },
                { label: "Reveal Secret", isTarget: true },
              ],
            },
            { label: "System Info" },
          ],
        },
        { label: "Shut Down..." },
      ],
    },
    {
      label: "Applications",
      items: [
        { label: "Browser" },
        { label: "Text Editor" },
        {
          label: "Utilities",
          items: [{ label: "Calculator" }, { label: "Terminal" }],
        },
      ],
    },
    {
      label: "Network",
      items: [
        { label: "Connections" },
        { label: "Firewall Settings" },
        {
          label: "Proxies",
          items: [
            { label: "Configure Proxy" },
            { label: "Test Proxy" },
            { label: "Bypass List" },
          ],
        },
      ],
    },
    {
      label: "User",
      items: [
        { label: "Profile" },
        {
          label: "Settings",
          items: [
            { label: "Account" },
            { label: "Privacy" },
            { label: "Notifications" },
          ],
        },
        { label: "Logout" },
      ],
    },
  ];

  const handleMenuEnter = (menuLabel: string) => {
    setActiveMenu(menuLabel);
    setActiveSubmenu(null);
    setActiveSubSubmenu(null);
  };

  const handleMenuLeave = () => {
    if (!activeSubmenu && !activeSubSubmenu) {
      setActiveMenu(null);
    }
  };

  const handleSubmenuEnter = (submenuLabel: string) => {
    setActiveSubmenu(submenuLabel);
    setActiveSubSubmenu(null);
  };

  const handleSubmenuLeave = () => {
    if (!activeSubSubmenu) {
      setActiveSubmenu(null);
    }
  };

  const handleSubSubmenuEnter = (subSubmenuLabel: string) => {
    setActiveSubSubmenu(subSubmenuLabel);
  };

  const handleItemClick = (item: MenuItem) => {
    if (item.isTarget) {
      setMessage({
        text: `Congratulations! The completion password is: ${PASSWORD_MenuNavigatorHard}`,
        type: "success",
      });
      setCompleted(true);
    } else if (!item.items) {
      // Only set error if it's a leaf node and not the target
      setMessage({
        text: "Incorrect selection. Keep looking!",
        type: "error",
      });
    }
    // If it has items, clicking does nothing, navigation is via hover
  };

  if (completed) {
    recordSuccess();
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-red-500 via-orange-500 to-yellow-500">
        <div className="bg-white/90 backdrop-blur rounded-lg p-8 shadow-lg text-center">
          <h2 className="text-2xl font-bold text-green-600 mb-2">Success!</h2>
          <p className="text-xl">The password is:</p>
          <p className="text-2xl font-mono mt-2 font-bold">
            {PASSWORD_MenuNavigatorHard}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-gray-200">
      {/* Instructions Card */}
      <div className="flex justify-center mt-4">
        <div className="bg-white rounded-lg p-4 mb-4 max-w-xl shadow-sm">
          <h1 className="text-xl font-bold text-center mb-2 text-gray-800">
            Menu Navigator (Hard)
          </h1>
          <p className="text-center text-sm text-gray-600">
            Navigate through the complex and deeply nested menu bar to find the
            secret option. Click it to reveal the password! Mouse hover reveals
            submenus.
          </p>
        </div>
      </div>

      {/* Desktop Window Container */}
      <div className="flex justify-center w-full">
        <div className="w-[900px] h-[600px] bg-white border border-gray-400 rounded-lg overflow-hidden">
          {/* Menu Bar */}
          <div className="bg-gray-900 text-gray-200 w-full">
            <div className="flex">
              {menuItems.map((menu) => (
                <div
                  key={menu.label}
                  className="relative"
                  onMouseEnter={() => handleMenuEnter(menu.label)}
                  onMouseLeave={handleMenuLeave}
                >
                  <button
                    className={`px-4 py-2.5 text-sm hover:bg-gray-700 ${
                      activeMenu === menu.label ? "bg-gray-700" : ""
                    }`}
                  >
                    {menu.label}
                  </button>
                  {activeMenu === menu.label && menu.items && (
                    <div
                      className="absolute left-0 top-full bg-gray-900 min-w-[200px] z-50 rounded-b-md shadow-lg"
                      onMouseLeave={handleMenuLeave} // Allows moving to submenu
                    >
                      {menu.items.map((item) => (
                        <div
                          key={item.label}
                          className="relative"
                          onMouseEnter={() => handleSubmenuEnter(item.label)}
                          onMouseLeave={handleSubmenuLeave}
                        >
                          <button
                            className="block w-full text-left px-3.5 py-2 text-sm hover:bg-gray-700 flex justify-between items-center"
                            onClick={() => handleItemClick(item)}
                          >
                            {item.label}
                            {item.items && <span className="ml-3">▶</span>}
                          </button>
                          {activeSubmenu === item.label && item.items && (
                            <div
                              className="absolute left-full top-0 bg-gray-900 min-w-[200px] rounded-md shadow-lg"
                              onMouseLeave={handleSubmenuLeave} // Allows moving to sub-submenu
                            >
                              {item.items.map((subItem) => (
                                <div
                                  key={subItem.label}
                                  className="relative"
                                  onMouseEnter={() =>
                                    handleSubSubmenuEnter(subItem.label)
                                  }
                                  // onMouseLeave here would close it if mouse moves out slightly, so not needed for deepest level
                                >
                                  <button
                                    className="block w-full text-left px-3.5 py-2 text-sm hover:bg-gray-700 flex justify-between items-center"
                                    onClick={() => handleItemClick(subItem)}
                                  >
                                    {subItem.label}
                                    {subItem.items && (
                                      <span className="ml-3">▶</span>
                                    )}
                                  </button>
                                  {activeSubSubmenu === subItem.label &&
                                    subItem.items && (
                                      <div className="absolute left-full top-0 bg-gray-900 min-w-[200px] rounded-md shadow-lg">
                                        {subItem.items.map((subSubItem) => (
                                          <button
                                            key={subSubItem.label}
                                            className="block w-full text-left px-3.5 py-2 text-sm hover:bg-gray-700"
                                            onClick={() =>
                                              handleItemClick(subSubItem)
                                            }
                                          >
                                            {subSubItem.label}
                                          </button>
                                        ))}
                                      </div>
                                    )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Desktop Background Area */}
          <div className="h-full bg-gradient-to-br from-red-500 via-orange-500 to-yellow-500 flex items-center justify-center">
            {message.text && (
              <div className="bg-white/90 backdrop-blur rounded-lg p-5 max-w-lg shadow-xl">
                <p
                  className={`text-lg font-semibold ${
                    message.type === "error" ? "text-red-700" : "text-green-700"
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

export default MenuNavigatorHard;
