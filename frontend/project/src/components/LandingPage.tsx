//import React from 'react';
import { Link } from 'react-router-dom';
import {
  Microscope,
  TrendingUp,
  DollarSign,
  Zap,
  Clock,
  ShieldCheck
} from 'lucide-react';

const features = [
  {
    icon: <Microscope className="h-8 w-8 text-blue-400" />,
    title: "Advanced O-Ring Defect Detection",
    description: "Cutting-edge AI technology for precise identification of manufacturing defects in O-rings with exceptional accuracy."
  },
  {
    icon: <TrendingUp className="h-8 w-8 text-blue-400" />,
    title: "Maximized Processing Volume",
    description: "High-throughput analysis capability enabling rapid inspection of large quantities of O-rings simultaneously."
  },
  {
    icon: <DollarSign className="h-8 w-8 text-blue-400" />,
    title: "Increased Profitability",
    description: "Reduce waste and quality control costs while improving production efficiency and customer satisfaction."
  },
  {
    icon: <Zap className="h-8 w-8 text-blue-400" />,
    title: "Real-Time Analysis",
    description: "Instant defect detection results powered by state-of-the-art machine learning models."
  },
  {
    icon: <Clock className="h-8 w-8 text-blue-400" />,
    title: "Time Efficiency",
    description: "Dramatically reduce inspection time compared to manual quality control processes."
  },
  {
    icon: <ShieldCheck className="h-8 w-8 text-blue-400" />,
    title: "Quality Assurance",
    description: "Maintain consistent product quality with automated inspection systems."
  }
];

const LandingPage = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold text-white mb-4 tracking-tight">
          AI-Powered Defect Detection for{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-200">
            Manufacturing Excellence
          </span>
        </h1>
        <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
          Revolutionize your quality control process with advanced machine learning technology
        </p>
        <Link
          to="/detect"
          className="inline-flex items-center px-8 py-4 border border-transparent text-lg font-medium rounded-md text-white bg-blue-600 hover:bg-blue-500 transition-all hover:scale-105 active:scale-95"
        >
          Start Detection
        </Link>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {features.map((feature, index) => (
          <div
            key={index}
            className="feature-card bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-blue-500/10 hover:border-blue-500/30 transition-all"
          >
            <div className="mb-4">{feature.icon}</div>
            <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
            <p className="text-gray-300">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default LandingPage;