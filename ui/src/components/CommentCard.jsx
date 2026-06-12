import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Eye, EyeOff, Clock } from 'lucide-react';

const CommentCard = ({ comment }) => {
    const isToxic = comment.label === 'Hate Speech' || comment.label === 'Offensive Language' || comment.label === 'Abusive' || comment.label === 'Hate';
    const [blurred, setBlurred] = useState(isToxic);

    const getBadgeColor = () => {
        switch (comment.label) {
            case 'Hate Speech':
            case 'Abusive':
            case 'Hate':
                return 'bg-red-500/20 text-red-500 border-red-500/50';
            case 'Offensive Language':
                return 'bg-orange-500/20 text-orange-500 border-orange-500/50';
            default:
                return 'bg-green-500/20 text-green-500 border-green-500/50';
        }
    };

    const formatTime = (seconds) => {
        if (seconds === null || seconds === undefined) return null;
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const timeString = comment.start !== undefined ? `${formatTime(comment.start)} - ${formatTime(comment.end)}` : null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`p-4 rounded-xl border border-gray-800 bg-gray-900/50 backdrop-blur-sm mb-3 hover:border-gray-700 transition-colors ${isToxic ? 'border-l-4 border-l-red-500' : 'border-l-4 border-l-green-500'}`}
        >
            <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                    <span className="font-bold text-gray-300 text-sm">{comment.author}</span>
                    {timeString && (
                        <span className="flex items-center gap-1 text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded border border-gray-700">
                            <Clock size={10} />
                            {timeString}
                        </span>
                    )}
                    <span className={`px-2 py-0.5 rounded-full text-xs border ${getBadgeColor()}`}>
                        {comment.label} ({Math.round(comment.confidence * 100)}%)
                    </span>
                </div>

                {isToxic && (
                    <button
                        onClick={() => setBlurred(!blurred)}
                        className="text-gray-500 hover:text-white transition-colors p-1"
                    >
                        {blurred ? <Eye size={16} /> : <EyeOff size={16} />}
                    </button>
                )}
            </div>

            <div className={`text-gray-200 text-sm leading-relaxed relative ${blurred ? 'blur-sm select-none' : ''}`}>
                {comment.text}
            </div>
        </motion.div>
    );
};

export default CommentCard;
