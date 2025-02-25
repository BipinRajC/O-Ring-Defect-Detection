import { Link } from 'react-router-dom';
import { CircuitBoard } from 'lucide-react';
// @ts-ignore
const Navbar = () => {
  return (
    <nav className="bg-gray-900/50 backdrop-blur-md border-b border-blue-500/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2 group">
              <CircuitBoard className="h-8 w-8 text-blue-400 group-hover:text-blue-300 transition-colors" />
              <span className="font-bold text-xl text-white group-hover:text-blue-300 transition-colors">AI Defect Detection</span>
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-gray-300 hover:text-blue-300 px-3 py-2 rounded-md transition-colors">
              Home
            </Link>
            <Link to="/about" className="text-gray-300 hover:text-blue-300 px-3 py-2 rounded-md transition-colors">
              About
            </Link>
            <Link
              to="/detect"
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-500 transition-all hover:scale-105 active:scale-95"
            >
              Detect Now
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;