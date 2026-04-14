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
        if (filter === 'toxic') return c.label === 'Abusive';
        if (filter === 'safe') return c.label === 'Not Abusive' || c.label === 'Neither';
        return true;
    });

    const toxicityPercentage = data.total_comments > 0
        ? Math.round((data.toxic_count / data.total_comments) * 100)
        : 0;

    const audioToxicityPercentage = data.audio_chunks?.length > 0
        ? Math.round((data.toxic_audio_count / data.audio_chunks.length) * 100)
        : 0;

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

            {/* Audio Analysis Stats */}
            {data.audio_chunks && data.audio_chunks.length > 0 && (
                <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 mb-8">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-4">Video Audio Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 bg-blue-500/10 rounded-xl flex items-center justify-center border border-blue-500/20">
                                <span className="text-2xl font-bold text-blue-500">{data.audio_chunks.length}</span>
                            </div>
                            <div>
                                <div className="text-gray-400 text-sm uppercase">Total Segments</div>
                                <div className="text-white">Processed Speech</div>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className={`w-16 h-16 rounded-xl flex items-center justify-center border ${data.toxic_audio_count > 0 ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20'}`}>
                                <span className={`text-2xl font-bold ${data.toxic_audio_count > 0 ? 'text-red-500' : 'text-green-500'}`}>
                                    {audioToxicityPercentage}%
                                </span>
                            </div>
                            <div>
                                <div className="text-gray-400 text-sm uppercase">Audio Toxicity</div>
                                <div className="text-white">{data.toxic_audio_count} Abusive Segments</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}


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
                    <div className="text-center py-20 text-gray-500 border border-gray-800 rounded-xl border-dashed">
                        No comments found for this filter.
                    </div>
                )}
            </div>

            {/* Video Audio Chunks List */}
            {data.audio_chunks && data.audio_chunks.length > 0 && (
                <div className="mt-12">
                    <h3 className="text-gray-400 text-sm font-semibold uppercase tracking-wider mb-6 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span> Transcribed Speech Segments
                    </h3>
                    <div className="space-y-4">
                        {data.audio_chunks.map((chunk, idx) => (
                            <div key={`audio-${idx}`} className={`p-4 rounded-xl border ${chunk.label === 'Abusive' ? 'bg-red-500/5 border-red-500/20' : 'bg-gray-800/30 border-gray-700/50'}`}>
                                <div className="flex flex-col md:flex-row gap-4 md:items-start justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <span className="text-xs font-mono bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
                                                {chunk.timestamp[0].toFixed(1)}s - {chunk.timestamp[1] ? chunk.timestamp[1].toFixed(1) + 's' : 'End'}
                                            </span>
                                            {chunk.label === 'Abusive' ? (
                                                <span className="bg-red-500/20 text-red-400 text-xs px-2 py-1 rounded-full border border-red-500/30 flex items-center gap-1"><AlertOctagon size={12} /> Abusive Speech</span>
                                            ) : (
                                                <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded-full border border-green-500/30 flex items-center gap-1"><ShieldCheck size={12} /> Safe</span>
                                            )}
                                        </div>
                                        <p className="text-gray-200 text-lg">"{chunk.text}"</p>
                                    </div>
                                    <div className="text-xs text-gray-500 font-mono self-end md:self-start bg-black/20 px-2 py-1 rounded border border-gray-700/50">
                                        Conf: {(chunk.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AnalysisDashboard;
