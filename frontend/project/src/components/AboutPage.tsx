import { Brain, Image as ImageIcon, LineChart } from 'lucide-react';
import industryImage from './industry.png';

const technologies = [
  {
    title: "Machine Learning",
    icon: <Brain className="h-8 w-8 text-blue-400" />,
    items: [
      {
        subtitle: "Deep Learning Frameworks",
        list: ["PyTorch", "TensorFlow", "Keras"]
      },
      {
        subtitle: "Neural Network Architectures",
        list: ["ResNet18", "Autoencoder", "Transfer Learning"]
      }
    ]
  },
  {
    title: "Computer Vision",
    icon: <ImageIcon className="h-8 w-8 text-blue-400" />,
    items: [
      {
        subtitle: "Image Processing",
        list: ["OpenCV", "PIL (Python Imaging Library)"]
      },
      {
        subtitle: "Preprocessing Techniques",
        list: ["Image Normalization", "Resize & Augmentation", "Color Space Conversion"]
      }
    ]
  },
  {
    title: "Data Science",
    icon: <LineChart className="h-8 w-8 text-blue-400" />,
    items: [
      {
        subtitle: "Libraries",
        list: ["NumPy", "Pandas", "Scikit-learn"]
      },
      {
        subtitle: "Statistical Analysis",
        list: ["Metrics Calculation", "Performance Evaluation"]
      }
    ]
  }
];

const benefits = [
  {
    icon: "📊",
    title: "Precision Detection",
    description: "Advanced AI algorithms detect defects with over 92% accuracy"
  },
  {
    icon: "⏱️",
    title: "Speed & Efficiency",
    description: "Process up to 500 O-rings per hour, dramatically reducing inspection time"
  },
  {
    icon: "💰",
    title: "Cost Reduction",
    description: "Minimize waste and quality control expenses through intelligent detection"
  },
  {
    icon: "🔬",
    title: "Continuous Learning",
    description: "Machine learning models improve with each inspection, enhancing accuracy"
  }
];

const AboutPage = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-white mb-4">
          About Our AI-Powered Defect Detection System
        </h1>
      </div>

      {/* Technology Stack */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-white mb-8 flex items-center">
          <Brain className="h-8 w-8 text-blue-400 mr-2" />
          Technology Stack
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {technologies.map((tech, index) => (
            <div key={index} className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-blue-500/10">
              <div className="flex items-center mb-4">
                {tech.icon}
                <h3 className="text-xl font-semibold text-white ml-2">{tech.title}</h3>
              </div>
              {tech.items.map((item, itemIndex) => (
                <div key={itemIndex} className="mb-4">
                  <h4 className="text-blue-400 font-medium mb-2">{item.subtitle}</h4>
                  <ul className="list-disc list-inside text-gray-300 space-y-1">
                    {item.list.map((listItem, listIndex) => (
                      <li key={listIndex}>{listItem}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Mission */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-white mb-8">Our Mission</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-1">
            <img
              src={industryImage}
              alt="AI in Manufacturing"
              className="rounded-lg w-full h-full object-cover"
            />
          </div>
          <div className="md:col-span-2 bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-blue-500/10">
            <h3 className="text-2xl font-bold text-white mb-4">Transforming Manufacturing Quality Control</h3>
            <p className="text-gray-300 mb-4">Our mission is to revolutionize industrial quality assurance by:</p>
            <ul className="list-disc list-inside text-gray-300 space-y-2 mb-4">
              <li>Leveraging cutting-edge AI technologies</li>
              <li>Reducing human error in defect detection</li>
              <li>Increasing production efficiency</li>
              <li>Minimizing waste and economic losses</li>
            </ul>
            <p className="text-blue-400 italic">
              Vision: Create intelligent, autonomous quality control systems that set new standards in manufacturing precision.
            </p>
          </div>
        </div>
      </div>

      {/* Benefits */}
      <div>
        <h2 className="text-3xl font-bold text-white mb-8">Key Benefits</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {benefits.map((benefit, index) => (
            <div
              key={index}
              className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-blue-500/10 text-center"
            >
              <div className="text-4xl mb-4">{benefit.icon}</div>
              <h3 className="text-xl font-semibold text-white mb-2">{benefit.title}</h3>
              <p className="text-gray-300">{benefit.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AboutPage;