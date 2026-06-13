import React, { useState } from 'react';
import axios from 'axios';
import SearchBar from './components/SearchBar';
import AnalysisDashboard from './components/AnalysisDashboard';
import { Github as GithubIcon, Upload as UploadIcon, FileAudio as AudioIcon, FileVideo as VideoIcon } from 'lucide-react';

function App() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const getApiUrl = () => {
    const isColab = window.location.hostname.includes('colab.dev');
    return isColab 
      ? window.location.origin.replace('5173', '8000') 
      : (import.meta.env.VITE_API_URL || 'http://localhost:8000');
  };

  const handleAnalyze = async (url) => {
    setIsLoading(true);
    setError('');
    setData(null);

    try {
      const apiUrl = getApiUrl();
      console.log("Connecting to API at:", apiUrl);
      
      const response = await axios.post(`${apiUrl}/analyze-video`, {
        url: url,
        limit: 100
      });
      setData(response.data);
    } catch (err) {
      console.error(err);
      const host = window.location.hostname;
      setError(`Connection Error: Could not reach backend at ${import.meta.env.VITE_API_URL || 'http://localhost:8000'}. Please ensure python -m src.api is running.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsLoading(true);
    setError('');
    setData(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = getApiUrl();
      const response = await axios.post(`${apiUrl}/analyze-file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });
      setData(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to analyze file. Ensure it is a valid video/audio file.');
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white selection:bg-blue-500 selection:text-white pb-20">
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
      </div>

      <div className="relative z-10 px-4 py-8 max-w-7xl mx-auto">
        <header className="flex justify-between items-center mb-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-tr from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <ShieldIcon />
            </div>
            <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              HG Multimodal
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <a href="https://github.com/atanu-m/HateGuard" target="_blank" rel="noreferrer" className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white">
              <GithubIcon size={24} />
            </a>
          </div>
        </header>

        <div className="text-center mb-12">
          <h2 className="text-5xl md:text-7xl font-black tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500 uppercase">
            Multimodal <br />
            <span className="text-blue-500">Toxicity Detection</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-10 leading-relaxed font-medium">
            AI-powered toxicity detection for YouTube and offline media in <span className="text-blue-400">100+ languages</span>.
          </p>

          <div className="flex flex-col items-center gap-6">
            <SearchBar onAnalyze={handleAnalyze} isLoading={isLoading} />
            
            <div className="flex items-center gap-4 text-gray-500">
              <div className="h-px w-16 bg-gray-800"></div>
              <span className="text-sm font-bold uppercase tracking-widest">OR UPLOAD</span>
              <div className="h-px w-16 bg-gray-800"></div>
            </div>

            <label className="group relative cursor-pointer">
              <input type="file" className="hidden" onChange={handleFileUpload} accept="video/*,audio/*" />
              <div className="flex items-center gap-3 px-8 py-4 bg-gray-800 hover:bg-gray-750 border border-gray-700 rounded-2xl transition-all group-hover:border-blue-500/50 group-hover:shadow-xl group-hover:shadow-blue-500/10">
                {isLoading && uploadProgress > 0 && uploadProgress < 100 ? (
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <span className="font-bold text-blue-400">{uploadProgress}% Uploading...</span>
                  </div>
                ) : (
                  <>
                    <div className="flex gap-1 text-blue-400">
                      <VideoIcon size={20} />
                      <AudioIcon size={20} />
                    </div>
                    <span className="font-bold text-gray-200">Analyze Offline Video or Audio</span>
                    <UploadIcon className="text-gray-500 group-hover:text-blue-400 transition-colors" size={20} />
                  </>
                )}
              </div>
            </label>
          </div>
        </div>

        {error && (
          <div className="max-w-md mx-auto mb-8 p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-200 text-center animate-in fade-in slide-in-from-top-4">
            {error}
          </div>
        )}

        <AnalysisDashboard data={data} />
        
        {/* Version Tag */}
        <div className="fixed bottom-4 right-4 text-[10px] text-gray-600 font-mono">
          v1.2 - MULTIMODAL_UPDATED
        </div>
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
