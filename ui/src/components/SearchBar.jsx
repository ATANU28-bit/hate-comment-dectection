import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const SearchBar = ({ onAnalyze, isLoading }) => {
    const [url, setUrl] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (url.trim()) {
            onAnalyze(url);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto mb-12">
            <motion.form
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                onSubmit={handleSubmit}
                className="relative group"
            >
                {/* Animated Glow Effect */}
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 rounded-full opacity-50 group-hover:opacity-100 blur transition duration-1000 group-hover:duration-200 animate-tilt"></div>

                <div className="relative flex items-center bg-gray-900 rounded-full p-2 shadow-2xl transition-colors">
                    <Search className="text-gray-400 ml-4" size={20} />
                    <input
                        type="text"
                        placeholder="Paste YouTube Video Link..."
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        className="w-full bg-transparent text-white px-4 py-3 focus:outline-none placeholder-gray-500 font-medium text-lg"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading}
                        className={`bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-full px-8 py-3 font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-blue-500/30 ${isLoading ? 'pr-4' : ''}`}
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="animate-spin" size={20} />
                                <span>Scanning...</span>
                            </>
                        ) : 'Analyze'}
                    </button>
                </div>
            </motion.form>
        </div>
    );
};

export default SearchBar;
