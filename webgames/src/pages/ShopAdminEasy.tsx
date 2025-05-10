import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { recordTaskView } from "../utils/analytics";
import { Product } from "./shopping/products"; // Assuming Product type is here

export const PASSWORD_ShopAdminEasy = "APPLE_PRICE_FIXED";
export const TASK_ID_ShopAdminEasy = "shop-admin-easy";

// Define a simple list of products for the easy version
const easyInitialProducts: Product[] = [
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

export default function ShopAdminEasy() {
  recordTaskView(TASK_ID_ShopAdminEasy, Date.now());
  const navigate = useNavigate();
  // Product state will be managed by ShopAdminEditEasy for this task's completion logic
  // This page primarily serves as an entry point and instruction display.
  const [displayedProducts] = useState<Product[]>(
    easyInitialProducts.slice(0, 3)
  ); // Show a few

  const targetProductName = "Apple";
  const targetPrice = 1.0;

  // Simulate finding the target product for display in instructions if needed
  const targetProductForDisplay = easyInitialProducts.find(
    (p) => p.name === targetProductName
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Shop Admin Panel (Easy)
        </h1>

        <div className="bg-green-50 p-6 rounded-lg shadow mb-8">
          <h2 className="text-xl font-bold mb-4">Challenge Instructions</h2>
          <p className="text-gray-700">
            Your task is to update the price of the
            <strong> {targetProductName} </strong>
            (currently ${targetProductForDisplay?.price.toFixed(2)}) to
            <strong> ${targetPrice.toFixed(2)}</strong>.
          </p>
          <p className="text-gray-700 mt-2">
            Click "Edit" on the correct product below to change its price.
          </p>
        </div>

        {/* Products Table - Simplified, no filters or search */}
        <div className="overflow-x-auto bg-white rounded-lg shadow">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Product
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {displayedProducts.map((product) => (
                <tr key={product.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">{product.emoji}</span>
                      <div>
                        <div className="font-medium text-gray-900">
                          {product.name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {product.description}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {product.category}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${product.price.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button
                      onClick={() =>
                        navigate(`/shop-admin-easy/edit/${product.id}`)
                      }
                      className="text-indigo-600 hover:text-indigo-900"
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* No pagination needed for the easy version */}
      </div>
    </div>
  );
}
