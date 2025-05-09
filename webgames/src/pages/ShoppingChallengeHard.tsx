import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
import { Product, products } from "./shopping/products";

interface ShoppingListItem {
  amount: string; // Natural description like "2 packs" or "24 eggs"
}

// Hard Shopping List:
// - 2 tropical fish (ID 6, Price 12.99 each) -> 2 * 12.99 = 25.98
// - 1 lobster (ID 11, Price 35.99)
// - 8 bananas (ID 66, "Bananas (bunch of six)" Price 2.89, and 2x "Single banana" ID 14, Price 0.59 each) -> 2.89 + (2 * 0.59) = 2.89 + 1.18 = 4.07
// - 2 bunches of grapes (ID 19, Price 3.99 each, assuming "bunch" means one unit of grapes) -> 2 * 3.99 = 7.98
// - 3 heads of broccoli (ID 25, Price 2.49 each) -> 3 * 2.49 = 7.47
// - 4 bell peppers (ID 33, Price 1.49 each) -> 4 * 1.49 = 5.96
// - 5 potatoes (ID 28, Price 0.79 each) -> 5 * 0.79 = 3.95
// - 2 blocks of cheese (ID 35, Price 4.99 each) -> 2 * 4.99 = 9.98
// - 24 eggs (2 dozen) (ID 36, "Eggs (12x)" Price 3.99 each pack) -> 2 * 3.99 = 7.98
// - 1 Coffee Maker (ID 54, Price 59.99)
// - 1 Air Fryer (ID 55, Price 79.99)
// Expected Total: 25.98 + 35.99 + 4.07 + 7.98 + 7.47 + 5.96 + 3.95 + 9.98 + 7.98 + 59.99 + 79.99 = 249.34

const shoppingList: ShoppingListItem[] = [
  { amount: "2 tropical fish" },
  { amount: "1 lobster" },
  { amount: "8 bananas (hint: look for bunches and singles for best price)" },
  { amount: "2 bunches of grapes" },
  { amount: "3 heads of broccoli" },
  { amount: "4 bell peppers" },
  { amount: "5 potatoes" },
  { amount: "2 blocks of cheese" },
  { amount: "24 eggs (2 dozen)" },
  { amount: "1 Coffee Maker (because why not?)" },
  { amount: "1 Air Fryer (for healthy snacks!)" },
];

const expectedTotal = 249.34;

interface CartItem extends Product {
  quantity: number;
}

export const PASSWORD_ShoppingChallengeHard = "HARD_SHOPPER_999";
export const TASK_ID_ShoppingChallengeHard = "shopping-challenge-hard";

const ShoppingChallengeHard: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ShoppingChallengeHard);
  const [cart, setCart] = useState<CartItem[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [activeSearchTerm, setActiveSearchTerm] = useState("");
  const [priceGuess, setPriceGuess] = useState("");
  const [message, setMessage] = useState("");
  const [password, setPassword] = useState("");
  const itemsPerPage = 6; // Fewer items per page for more scrolling
  const [isCartOpen, setIsCartOpen] = useState(false);
  const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);

  const categories = ["All", ...new Set(products.map((p) => p.category))];

  const filteredProducts =
    selectedCategory === "All"
      ? products
      : products.filter((p) => p.category === selectedCategory);

  const searchFilteredProducts = activeSearchTerm
    ? filteredProducts.filter(
        (p) =>
          p.name.toLowerCase().includes(activeSearchTerm.toLowerCase()) ||
          p.description.toLowerCase().includes(activeSearchTerm.toLowerCase())
      )
    : filteredProducts;

  const totalPages = Math.ceil(searchFilteredProducts.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const displayedProducts = searchFilteredProducts.slice(
    startIndex,
    startIndex + itemsPerPage
  );

  const addToCart = (product: Product) => {
    setCart((prevCart) => {
      const existingItem = prevCart.find((item) => item.id === product.id);
      if (existingItem) {
        return prevCart.map((item) =>
          item.id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      return [...prevCart, { ...product, quantity: 1 }];
    });
  };

  const removeFromCart = (productId: number) => {
    setCart((prevCart) => {
      const existingItem = prevCart.find((item) => item.id === productId);
      if (existingItem && existingItem.quantity > 1) {
        return prevCart.map((item) =>
          item.id === productId
            ? { ...item, quantity: item.quantity - 1 }
            : item
        );
      }
      return prevCart.filter((item) => item.id !== productId);
    });
  };

  const checkPrice = () => {
    const guessedPrice = parseFloat(priceGuess);
    // Increased tolerance for hard version due to complexity
    if (Math.abs(guessedPrice - expectedTotal) < 0.05) {
      setPassword(PASSWORD_ShoppingChallengeHard);
      setMessage("Amazing! You've mastered the hard shopping challenge!");
      recordSuccess();
    } else {
      setMessage(
        "That's not the correct total. This one is tricky, check your math and items!"
      );
    }
  };

  const handleSearch = () => {
    setActiveSearchTerm(searchTerm);
    setCurrentPage(1);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8 text-center">
        Shopping Challenge (Hard)
      </h1>
      <p className="text-center mb-6">
        Add the required items to your cart and calculate the CHEAPEST possible
        total price to complete the challenge! This one is tough.
      </p>

      {/* Shopping List */}
      <div className="mb-8 p-6 bg-yellow-100 border-l-4 border-yellow-500 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Shopping List (Hard Mode!)</h2>
        <textarea
          value={shoppingList.map((item) => item.amount).join("\n")}
          className="w-full h-40 p-2 border rounded mb-4 bg-yellow-50"
          readOnly
        />
        <p className="text-sm text-yellow-700">
          Note: Ensure you select the most cost-effective options for items like
          'bananas' or 'eggs' where multiple packaging options might exist.
        </p>
      </div>

      {/* Category Filter, Search, and Cart */}
      <div className="mb-6 flex flex-wrap justify-between items-center gap-4">
        <div className="flex flex-wrap gap-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => {
                setSelectedCategory(category);
                setCurrentPage(1);
              }}
              className={`px-4 py-2 rounded ${
                selectedCategory === category
                  ? "bg-indigo-500 text-white"
                  : "bg-gray-200"
              }`}
            >
              {category}
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search products..."
            className="px-4 py-2 border rounded"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSearch();
              }
            }}
          />
          <button
            onClick={handleSearch}
            className="px-4 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600"
          >
            Search
          </button>
        </div>

        <div className="relative">
          <button
            onClick={() => setIsCartOpen(!isCartOpen)}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-700 hover:text-gray-900 border border-gray-300 hover:border-gray-400 bg-white"
          >
            <span className="text-xl">🛒</span>
            <span>Cart ({totalItems})</span>
          </button>

          {/* Cart Modal */}
          {isCartOpen && (
            <>
              {/* Backdrop */}
              <div
                className="fixed inset-0 bg-black bg-opacity-50 z-40"
                onClick={() => setIsCartOpen(false)}
              />

              {/* Cart Content */}
              <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-xl z-50 max-h-[80vh] overflow-y-auto">
                <div className="p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold">Shopping Cart</h2>
                    <button
                      onClick={() => setIsCartOpen(false)}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      ✕
                    </button>
                  </div>
                  {cart.length === 0 ? (
                    <p>Your cart is empty</p>
                  ) : (
                    <div className="space-y-4">
                      {cart.map((item) => (
                        <div
                          key={item.id}
                          className="flex items-center justify-between border-b pb-2"
                        >
                          <div className="flex items-center">
                            <span className="text-2xl mr-2">{item.emoji}</span>
                            <span>{item.name}</span>
                          </div>
                          <div className="flex items-center space-x-4">
                            <span>
                              ${(item.price * item.quantity).toFixed(2)}
                            </span>
                            <div className="flex items-center space-x-2">
                              <button
                                onClick={() => removeFromCart(item.id)}
                                className="px-2 py-1 bg-red-500 text-white rounded"
                              >
                                -
                              </button>
                              <span>{item.quantity}</span>
                              <button
                                onClick={() => addToCart(item)}
                                className="px-2 py-1 bg-green-500 text-white rounded"
                              >
                                +
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                      <div className="pt-4 mt-4 border-t">
                        <div className="flex justify-between items-center">
                          <span className="text-xl font-bold">Total:</span>
                          <span className="text-xl font-bold">
                            $
                            {cart
                              .reduce(
                                (sum, item) => sum + item.price * item.quantity,
                                0
                              )
                              .toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Product Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {" "}
        {/* Adjusted for potentially more scrolling */}
        {displayedProducts.map((product) => (
          <div
            key={product.id}
            className="border p-4 rounded-lg shadow-md flex flex-col items-center"
          >
            <span className="text-6xl mb-2">{product.emoji}</span>
            <h3 className="text-xl font-semibold">{product.name}</h3>
            <p className="text-gray-600 mb-2 text-sm">{product.description}</p>
            <p className="text-lg font-bold mb-2">
              ${product.price.toFixed(2)}
            </p>
            <button
              onClick={() => addToCart(product)}
              className="bg-indigo-500 text-white px-4 py-2 rounded hover:bg-indigo-600 w-full mt-auto"
            >
              Add to Cart
            </button>
          </div>
        ))}
      </div>

      {/* Pagination */}
      <div className="flex justify-center space-x-4 mb-8">
        <button
          onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
          className="px-4 py-2 rounded bg-gray-200 disabled:opacity-50"
        >
          Previous
        </button>
        <span className="px-4 py-2">
          Page {currentPage} of {totalPages}
        </span>
        <button
          onClick={() =>
            setCurrentPage((prev) => Math.min(prev + 1, totalPages))
          }
          disabled={currentPage === totalPages}
          className="px-4 py-2 rounded bg-gray-200 disabled:opacity-50"
        >
          Next
        </button>
      </div>

      {/* Price Guess Form */}
      <div className="max-w-md mx-auto">
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Enter the total price for all required items:
          </label>
          <input
            type="number"
            step="0.01"
            value={priceGuess}
            onChange={(e) => setPriceGuess(e.target.value)}
            className="w-full px-4 py-2 border rounded"
            placeholder="Enter total price"
          />
        </div>
        <button
          onClick={checkPrice}
          className="w-full bg-indigo-500 text-white px-4 py-2 rounded hover:bg-indigo-600"
        >
          Check Price
        </button>
        {message && (
          <p className="mt-4 text-center font-semibold">
            {message}
            {password && (
              <span className="block mt-2 text-green-600">
                Password: {password}
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
};

export default ShoppingChallengeHard;
