import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { recordTaskView } from "../utils/analytics";
import {
  Product,
  products as initialProductsFromDependency,
} from "./shopping/products";

export const PASSWORD_ShopAdminHard = "VIP_GRINDER_DEAL";
export const TASK_ID_ShopAdminHard = "shop-admin-hard";

export default function ShopAdminHard() {
  recordTaskView(TASK_ID_ShopAdminHard, Date.now());
  const navigate = useNavigate();

  // For this version, we assume initialProductsFromDependency is the source of truth
  // and ShopAdminEditHard will handle its modification if necessary for password check.
  // The display here will show the state at load time.
  const [products] = useState<Product[]>(initialProductsFromDependency);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [searchTerm, setSearchTerm] = useState("");
  const itemsPerPage = 10;

  const targetProductName = "Coffee Grinder";
  const originalPrice = 25.5;
  const discountPercentage = 0.15;
  const newPrice = parseFloat(
    (originalPrice * (1 - discountPercentage)).toFixed(2)
  ); // 21.68

  // Recalculate categories based on the current products state
  const categories = ["All", ...new Set(products.map((p) => p.category))];

  const filteredProducts =
    selectedCategory === "All"
      ? products
      : products.filter((p) => p.category === selectedCategory);

  const searchFilteredProducts = searchTerm
    ? filteredProducts.filter(
        (p) =>
          p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          p.description.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : filteredProducts;

  const totalPages = Math.ceil(searchFilteredProducts.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const displayedProducts = searchFilteredProducts.slice(
    startIndex,
    startIndex + itemsPerPage
  );

  // handleDelete might be part of a more complex hard task, but not this one.
  // const handleDelete = (productId: number) => {
  //   setProducts((prevProducts) =>
  //     prevProducts.filter((p) => p.id !== productId)
  //   );
  // };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Shop Admin Panel (Hard)
        </h1>

        <div className="bg-red-50 p-6 rounded-lg shadow mb-8">
          <h2 className="text-xl font-bold mb-4">Challenge Instructions</h2>
          <p className="text-gray-700">
            The <strong>{targetProductName}</strong> is currently priced at $
            <strong>{originalPrice.toFixed(2)}</strong>. A VIP customer is
            eligible for a 15% discount.
          </p>
          <p className="text-gray-700 mt-1">
            You need to find the {targetProductName}, calculate its new price
            after the 15% discount (rounded to the nearest cent), and update it
            to this new price (which should be $
            <strong>{newPrice.toFixed(2)}</strong>).
          </p>
        </div>

        {/* Filters */}
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
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              placeholder="Search products..."
              className="px-4 py-2 border rounded"
            />
          </div>
        </div>

        {/* Products Table */}
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
                    <div className="flex gap-3">
                      <button
                        onClick={() =>
                          navigate(`/shop-admin-hard/edit/${product.id}`)
                        }
                        className="text-indigo-600 hover:text-indigo-900"
                      >
                        Edit
                      </button>
                      {/* Delete button is not part of this specific hard task
                      <button
                        onClick={() => handleDelete(product.id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        Delete
                      </button> */}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex justify-center space-x-4 mt-6">
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
      </div>
    </div>
  );
}
