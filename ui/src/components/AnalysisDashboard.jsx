import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { ShieldCheck, AlertOctagon, Link, FileText } from 'lucide-react';
import CommentCard from './CommentCard';

const AnalysisDashboard = ({ data }) => {
    const [filter, setFilter] = useState('all'); // all, toxic, safe

    if (!data) return null;

    const chartData = [
        { name: 'Safe', value: data.safe_count, color: '#22c55e' },
        { name: 'Toxic', value: data.toxic_count, color: '#ef4444' },
    ];

    const analysisItems = data.analysis || [];

    const filteredItems = analysisItems.filter(c => {
        const isToxicLabel = c.label === 'Hate Speech' || c.label === 'Offensive Language' || c.label === 'Abusive' || c.label === 'Hate';
        if (filter === 'toxic') return isToxicLabel;
        if (filter === 'safe') return !isToxicLabel;
        return true;
    });

    const totalCount = data.total_segments || data.total_comments || 0;
    const toxicityPercentage = Math.round((data.toxic_count / totalCount) * 100) || 0;
    const isYoutube = data.source && (data.source.includes('youtube.com') || data.source.includes('youtu.be'));

    return (
        <div className="w-full max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-10 duration-700">

            {/* Stats Header */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {/* Card 1: Toxicity Score */}
                <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 relative overflow-hidden group hover:border-gray-600 transition-colors">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">Toxicity Score</h3>
                    <div className="flex items-end gap-2">
                        <span className={`text-5xl font-black ${toxicityPercentage > 20 ? 'text-red-500' : 'text-green-500'}`}>
                            {toxicityPercentage}%
                        </span>
                        <span className="text-gray-500 mb-2">toxicity detected</span>
                    </div>
                    <div className="absolute right-4 top-4 opacity-20 group-hover:scale-110 transition-transform">
                        {toxicityPercentage > 20 ? <AlertOctagon size={48} className="text-red-500" /> : <ShieldCheck size={48} className="text-green-500" />}
                    </div>
                </div>

                {/* Card 2: Chart */}
                <div className="bg-gray-800/50 rounded-2xl p-4 border border-gray-700 flex items-center justify-center">
                    <div className="relative w-24 h-24">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={chartData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={30}
                                    outerRadius={45}
                                    paddingAngle={5}
                                    dataKey="value"
                                    stroke="none"
                                >
                                    {chartData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', borderRadius: '8px', color: '#fff' }}
                                    itemStyle={{ color: '#fff' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="ml-4 text-xs font-medium space-y-1">
                        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-green-500"></div> Safe: {data.safe_count}</div>
                        <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-red-500"></div> Toxic: {data.toxic_count}</div>
                    </div>
                </div>

                {/* Card 3: Source Info */}
                <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-2">Source Analysis</h3>
                    <div 
                        className={`text-sm truncate font-medium flex items-center gap-2 ${isYoutube ? 'text-blue-400 underline cursor-pointer hover:text-blue-300' : 'text-gray-300'}`}
                        onClick={() => isYoutube && window.open(data.source, '_blank')}
                    >
                        {isYoutube ? <Link size={14} /> : <FileText size={14} />}
                        {data.source}
                    </div>
                    <div className="mt-4 text-gray-400 text-sm">
                        Analyzed <span className="font-bold text-white">{totalCount}</span> segments
                    </div>
                </div>
            </div>

            {/* Filter Tabs */}
            <div className="flex flex-wrap gap-3 mb-6 bg-gray-900/40 p-1.5 rounded-2xl border border-gray-800/50 backdrop-blur-sm shadow-xl inline-flex">
                <button
                    onClick={() => setFilter('all')}
                    className={`px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${filter === 'all' ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20' : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'}`}
                >
                    Full Analysis
                </button>
                <button
                    onClick={() => setFilter('toxic')}
                    className={`px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${filter === 'toxic' ? 'bg-gradient-to-r from-red-600 to-red-500 text-white shadow-lg shadow-red-500/20' : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'}`}
                >
                    Flagged <span className="ml-2 bg-black/30 px-2 py-0.5 rounded-lg text-[10px]">{data.toxic_count}</span>
                </button>
                <button
                    onClick={() => setFilter('safe')}
                    className={`px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${filter === 'safe' ? 'bg-gradient-to-r from-green-600 to-green-500 text-white shadow-lg shadow-green-500/20' : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'}`}
                >
                    Verified Safe <span className="ml-2 bg-black/30 px-2 py-0.5 rounded-lg text-[10px]">{data.safe_count}</span>
                </button>
            </div>

            {/* Content List */}
            <div className="space-y-3 pb-20">
                {filteredItems.map((item, idx) => (
                    <CommentCard key={idx} comment={item} />
                ))}
                {filteredItems.length === 0 && (
                    <div className="text-center py-24 bg-gray-800/20 rounded-3xl border border-dashed border-gray-700">
                        <div className="text-gray-500 font-medium">No results found for this filter.</div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AnalysisDashboard;
