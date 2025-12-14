import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { ShieldCheck, AlertOctagon, Filter } from 'lucide-react';
import CommentCard from './CommentCard';

const AnalysisDashboard = ({ data }) => {
    const [filter, setFilter] = useState('all'); // all, toxic, safe

    if (!data) return null;

    const chartData = [
        { name: 'Safe', value: data.safe_count, color: '#22c55e' },
        { name: 'Toxic', value: data.toxic_count, color: '#ef4444' },
    ];

    const filteredComments = data.comments.filter(c => {
        if (filter === 'toxic') return c.label === 'Hate Speech' || c.label === 'Offensive Language';
        if (filter === 'safe') return c.label === 'Neither';
        return true;
    });

    const toxicityPercentage = Math.round((data.toxic_count / data.total_comments) * 100) || 0;

    return (
        <div className="w-full max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-10 duration-700">

            {/* Stats Header */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Card 1: Toxicity Score */}
                <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 relative overflow-hidden">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">Toxicity Score</h3>
                    <div className="flex items-end gap-2">
                        <span className={`text-5xl font-black ${toxicityPercentage > 50 ? 'text-red-500' : 'text-green-500'}`}>
                            {toxicityPercentage}%
                        </span>
                        <span className="text-gray-500 mb-2">of comments are toxic</span>
                    </div>
                    <div className="absolute right-4 top-4 opacity-20">
                        {toxicityPercentage > 50 ? <AlertOctagon size={48} /> : <ShieldCheck size={48} />}
                    </div>
                </div>

                {/* Card 2: Chart */}
                <div className="bg-gray-800/50 rounded-2xl p-4 border border-gray-700 flex items-center justify-center">
                    <ResponsiveContainer width="100%" height={100}>
                        <PieChart>
                            <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={30}
                                outerRadius={45}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                                itemStyle={{ color: '#fff' }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="ml-4 text-sm">
                        <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-green-500"></div> Safe: {data.safe_count}</div>
                        <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500"></div> Toxic: {data.toxic_count}</div>
                    </div>
                </div>

                {/* Card 3: Video Info */}
                <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">Video Analyzed</h3>
                    <div className="text-blue-400 text-sm truncate underline cursor-pointer" onClick={() => window.open(data.video_url, '_blank')}>
                        {data.video_url}
                    </div>
                    <div className="mt-4 text-gray-300">
                        Scanning <span className="font-bold text-white">{data.total_comments}</span> comments
                    </div>
                </div>
            </div>

            {/* Filter Tabs */}
            <div className="flex gap-4 mb-6 sticky top-4 z-10 bg-gray-900/80 backdrop-blur-md p-2 rounded-xl border border-gray-800 inline-flex">
                <button
                    onClick={() => setFilter('all')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === 'all' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30' : 'text-gray-400 hover:text-white'}`}
                >
                    All Comments
                </button>
                <button
                    onClick={() => setFilter('toxic')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === 'toxic' ? 'bg-red-600 text-white shadow-lg shadow-red-500/30' : 'text-gray-400 hover:text-white'}`}
                >
                    Toxic Only <span className="ml-2 bg-black/20 px-2 py-0.5 rounded-full text-xs">{data.toxic_count}</span>
                </button>
                <button
                    onClick={() => setFilter('safe')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${filter === 'safe' ? 'bg-green-600 text-white shadow-lg shadow-green-500/30' : 'text-gray-400 hover:text-white'}`}
                >
                    Safe Only <span className="ml-2 bg-black/20 px-2 py-0.5 rounded-full text-xs">{data.safe_count}</span>
                </button>
            </div>

            {/* Comment List */}
            <div className="space-y-2">
                {filteredComments.map((comment, idx) => (
                    <CommentCard key={idx} comment={comment} />
                ))}
                {filteredComments.length === 0 && (
                    <div className="text-center py-20 text-gray-500">
                        No comments found for this filter.
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalysisDashboard;
