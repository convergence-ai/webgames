import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { recordTaskView } from "../utils/analytics";
import { PASSWORD_ShopAdminEasy, TASK_ID_ShopAdminEasy } from "./ShopAdminEasy";
import { Product } from "./shopping/products"; // Assuming Product type is here

// Define the same simple list of products as in ShopAdminEasy.tsx
// In a real app, this might come from a shared service or context
const easyProducts: Product[] = [
  {
    id: 101,
    name: "Apple",
    emoji: "🍎",
    category: "Fruits",
    description: "A crisp red apple.",
    price: 1.25,
  },
  {
    id: 102,
    name: "Banana",
    emoji: "🍌",
    category: "Fruits",
    description: "A ripe yellow banana.",
    price: 0.75,
  },
  {
    id: 103,
    name: "Orange",
    emoji: "🍊",
    category: "Fruits",
    description: "A juicy orange.",
    price: 0.9,
  },
];

const TARGET_PRODUCT_ID_EASY = 101; // Apple's ID
const TARGET_PRICE_EASY = 1.0;

export default function ShopAdminEditEasy() {
  const { productId } = useParams<{ productId: string }>();
  const navigate = useNavigate();
  const [product, setProduct] = useState<Product | undefined>(undefined);
  const [price, setPrice] = useState("");
  const [passwordRevealed, setPasswordRevealed] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    recordTaskView(`${TASK_ID_ShopAdminEasy}-edit`, Date.now());
    const currentProduct = easyProducts.find((p) => p.id === Number(productId));
    if (currentProduct) {
      setProduct(currentProduct);
      setPrice(currentProduct.price.toFixed(2));
    } else {
      setError("Product not found.");
    }
  }, [productId]);

  const handleSave = () => {
    setError("");
    const newPrice = parseFloat(price);
    if (isNaN(newPrice)) {
      setError("Invalid price format. Please enter a number.");
      return;
    }

    if (
      product &&
      product.id === TARGET_PRODUCT_ID_EASY &&
      newPrice === TARGET_PRICE_EASY
    ) {
      // In a real app, you would also update the product list here or via a service
      setPasswordRevealed(true);
    } else if (product) {
      setError(
        `Incorrect action. Target: ${
          easyProducts.find((p) => p.id === TARGET_PRODUCT_ID_EASY)?.name
        } to $${TARGET_PRICE_EASY.toFixed(2)}. You tried to set ${
          product.name
        } to $${newPrice.toFixed(2)}`
      );
    } else {
      setError("An unexpected error occurred.");
    }
  };

  if (error && !product) {
    return (
      <div className="container mx-auto px-4 py-8 text-red-500">
        Error: {error}
      </div>
    );
  }

  if (!product) {
    return (
      <div className="container mx-auto px-4 py-8">Loading product...</div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-md mx-auto bg-white p-8 rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-6">Edit Product (Easy)</h1>
        <div className="mb-6">
          <h2 className="text-xl font-semibold">
            {product.emoji} {product.name}
          </h2>
          <p className="text-sm text-gray-600">{product.description}</p>
          <p className="text-sm text-gray-600">Category: {product.category}</p>
        </div>

        <div className="mb-4">
          <label
            htmlFor="price"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Price ($)
          </label>
          <input
            type="number"
            id="price"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
            step="0.01"
          />
        </div>

        {error && !passwordRevealed && (
          <p className="text-red-500 text-sm mb-4">{error}</p>
        )}

        {passwordRevealed ? (
          <div className="mt-6 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
            <h3 className="font-bold">Success!</h3>
            <p>
              Password:{" "}
              <span className="font-mono">{PASSWORD_ShopAdminEasy}</span>
            </p>
            <button
              onClick={() => navigate("/" + TASK_ID_ShopAdminEasy)}
              className="mt-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Back to Task
            </button>
          </div>
        ) : (
          <div className="flex justify-between items-center mt-6">
            <button
              onClick={handleSave}
              className="px-6 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Save Price
            </button>
            <button
              onClick={() => navigate("/" + TASK_ID_ShopAdminEasy)}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
