import React, { useState } from 'react';
import axios from 'axios';
import SearchBar from './components/SearchBar';
import AnalysisDashboard from './components/AnalysisDashboard';
import { Github, Youtube } from 'lucide-react';

function App() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async (url) => {
    setIsLoading(true);
    setError('');
    setData(null);

    try {
      // Connect to Backend (Uses environment variable or defaults to localhost)
      const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.post(`${apiUrl}/analyze-video`, {
        url: url,
        limit: 50 // Limit to 50 for quick demo, increase for prod
      });
      setData(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to analyze video. Make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white selection:bg-blue-500 selection:text-white">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px]"></div>
      </div>

      <div className="relative z-10 px-4 py-8 max-w-7xl mx-auto">
        {/* Header */}
        <header className="flex justify-between items-center mb-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-tr from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <ShieldIcon />
            </div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              HateGuard
            </h1>
          </div>
          <a href="#" className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white">
            <Github size={24} />
          </a>
        </header>

        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-5xl md:text-7xl font-black tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
            Clean Up the <br />
            <span className="text-blue-500">Comment Section</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Instantly analyze YouTube comments for hate speech and toxicity using advanced AI (DistilBERT).
          </p>

          <SearchBar onAnalyze={handleAnalyze} isLoading={isLoading} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="max-w-md mx-auto mb-8 p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-200 text-center animate-in fade-in slide-in-from-top-4">
            {error}
          </div>
        )}

        {/* Dashboard */}
        <AnalysisDashboard data={data} />

      </div>
    </div>
  );
}

const ShieldIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 22C12 22 20 18 20 12V5L12 2L4 5V12C4 18 12 22 12 22Z" fill="white" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export default App;
