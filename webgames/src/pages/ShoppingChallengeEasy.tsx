import React, { useState } from "react";
import { useTaskAnalytics } from "../utils/useTaskAnalytics";
import { Product, products } from "./shopping/products";

interface ShoppingListItem {
  amount: string; // Natural description like "2 packs" or "24 eggs"
}

// Easy Shopping List:
// - 1 tropical fish (ID 6, Price 12.99)
// - 3 bananas (using "Single banana" ID 14, Price 0.59 each) -> 3 * 0.59 = 1.77
// - 1 head of broccoli (ID 25, Price 2.49)
// Expected Total: 12.99 + 1.77 + 2.49 = 17.25
const shoppingList: ShoppingListItem[] = [
  { amount: "1 tropical fish" },
  { amount: "3 bananas" },
  { amount: "1 head of broccoli" },
];

const expectedTotal = 17.25;

interface CartItem extends Product {
  quantity: number;
}

export const PASSWORD_ShoppingChallengeEasy = "EASY_SHOPPER_001";
export const TASK_ID_ShoppingChallengeEasy = "shopping-challenge-easy";

const ShoppingChallengeEasy: React.FC = () => {
  const { recordSuccess } = useTaskAnalytics(TASK_ID_ShoppingChallengeEasy);
  const [cart, setCart] = useState<CartItem[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [searchTerm, setSearchTerm] = useState("");
  const [activeSearchTerm, setActiveSearchTerm] = useState("");
  const [priceGuess, setPriceGuess] = useState("");
  const [message, setMessage] = useState("");
  const [password, setPassword] = useState("");
  const itemsPerPage = 8;
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
    if (Math.abs(guessedPrice - expectedTotal) < 0.01) {
      setPassword(PASSWORD_ShoppingChallengeEasy);
      setMessage("Correct! You've completed the easy challenge!");
      recordSuccess();
    } else {
      setMessage("That's not the correct total. Try again!");
    }
  };

  const handleSearch = () => {
    setActiveSearchTerm(searchTerm);
    setCurrentPage(1);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8 text-center">
        Shopping Challenge (Easy)
      </h1>
      <p className="text-center mb-6">
        Add the required items to your cart and calculate the cheapest possible
        total price to complete the challenge!
      </p>

      {/* Shopping List */}
      <div className="mb-8 p-6 bg-yellow-50 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Shopping List</h2>
        <textarea
          value={shoppingList.map((item) => item.amount).join("\n")}
          className="w-full h-20 p-2 border rounded mb-4"
          readOnly
        />
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
                  ? "bg-blue-500 text-white"
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
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {displayedProducts.map((product) => (
          <div
            key={product.id}
            className="border p-4 rounded-lg shadow-md flex flex-col items-center"
          >
            <span className="text-6xl mb-2">{product.emoji}</span>
            <h3 className="text-xl font-semibold">{product.name}</h3>
            <p className="text-gray-600 mb-2">{product.description}</p>
            <p className="text-lg font-bold mb-2">
              ${product.price.toFixed(2)}
            </p>
            <button
              onClick={() => addToCart(product)}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
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
          className="w-full bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
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

export default ShoppingChallengeEasy;
