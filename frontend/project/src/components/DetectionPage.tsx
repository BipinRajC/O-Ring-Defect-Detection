import React, { useState } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';

const DetectionPage = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    setSelectedFile(file);
    // Redirect to Streamlit app after a short delay to show the "redirecting" message
    setTimeout(() => {
      window.location.href = 'http://localhost:8501';
    }, 1500);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-4">
          O-Ring Defect Detection
        </h1>
        <p className="text-gray-300">
          Upload an image of an O-ring for AI-powered defect analysis
        </p>
      </div>

      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all ${
          dragActive 
            ? "border-blue-400 bg-blue-900/30" 
            : "border-blue-500/30 bg-gray-800/30 hover:border-blue-500/50"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
          id="file-upload"
        />
        
        <div className="space-y-4">
          {selectedFile ? (
            <div className="space-y-4">
              <ImageIcon className="h-16 w-16 mx-auto text-blue-400" />
              <p className="text-lg font-medium text-white">{selectedFile.name}</p>
              <p className="text-blue-300">
                File selected. Redirecting to analysis...
              </p>
            </div>
          ) : (
            <>
              <Upload className="h-16 w-16 mx-auto text-blue-400" />
              <div className="space-y-2">
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer inline-flex items-center px-6 py-3 text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-500 transition-all hover:scale-105 active:scale-95"
                >
                  Choose a file
                </label>
                <p className="text-gray-300">or drag and drop</p>
              </div>
              <p className="text-gray-400">
                Supported formats: PNG, JPG, JPEG
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default DetectionPage;