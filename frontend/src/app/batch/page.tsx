'use client';

import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  Send, 
  Settings, 
  Play, 
  Pause, 
  XCircle, 
  CheckCircle2, 
  Clock, 
  AlertTriangle,
  ChevronRight
} from 'lucide-react';

interface BatchFile {
  filename: string;
  verdict: 'fake' | 'real';
  score: number;
  status: 'completed' | 'processing' | 'pending';
}

interface BatchJob {
  id: string;
  priority: 'low' | 'medium' | 'high';
  webhookUrl: string;
  totalFiles: number;
  completedCount: number;
  status: 'queued' | 'processing' | 'completed' | 'paused';
  files: BatchFile[];
  timestamp: string;
}

const INITIAL_JOBS: BatchJob[] = [
  {
    id: 'BATCH-2026-9284',
    priority: 'high',
    webhookUrl: 'https://api.enterprise.com/webhooks/deepfake',
    totalFiles: 8,
    completedCount: 8,
    status: 'completed',
    timestamp: '2026-06-12 10:14',
    files: [
      { filename: 'leak_01.mp4', verdict: 'fake', score: 0.94, status: 'completed' },
      { filename: 'press_shot_01.jpg', verdict: 'real', score: 0.02, status: 'completed' },
      { filename: 'memo_audio.wav', verdict: 'fake', score: 0.88, status: 'completed' },
      { filename: 'profile_face.png', verdict: 'fake', score: 0.72, status: 'completed' },
      { filename: 'speech_clip.mp4', verdict: 'real', score: 0.12, status: 'completed' },
      { filename: 'audio_briefing.mp3', verdict: 'real', score: 0.08, status: 'completed' },
      { filename: 'synthetic_ad.mp4', verdict: 'fake', score: 0.81, status: 'completed' },
      { filename: 'ceo_statement.jpg', verdict: 'real', score: 0.03, status: 'completed' },
    ]
  },
  {
    id: 'BATCH-2026-9285',
    priority: 'medium',
    webhookUrl: 'https://security.newsagency.org/alerts',
    totalFiles: 4,
    completedCount: 2,
    status: 'processing',
    timestamp: '2026-06-12 14:50',
    files: [
      { filename: 'wire_leak_raw.wav', verdict: 'fake', score: 0.91, status: 'completed' },
      { filename: 'staged_combat.mp4', verdict: 'fake', score: 0.85, status: 'completed' },
      { filename: 'unverified_avatar.png', verdict: 'real', score: 0.0, status: 'processing' },
      { filename: 'incoming_press_photo.jpg', verdict: 'real', score: 0.0, status: 'pending' },
    ]
  }
];

export default function BatchProcessing() {
  const [jobs, setJobs] = useState<BatchJob[]>(INITIAL_JOBS);
  const [urlsInput, setUrlsInput] = useState<string>('');
  const [priority, setPriority] = useState<'low' | 'medium' | 'high'>('medium');
  const [webhookUrl, setWebhookUrl] = useState<string>('');
  const [expandedJobId, setExpandedJobId] = useState<string | null>('BATCH-2026-9285');

  // Simulate progress updates for active processing jobs
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs(prevJobs => 
        prevJobs.map(job => {
          if (job.status === 'processing') {
            const nextCompleted = job.completedCount + 1;
            const updatedFiles = job.files.map((f, idx) => {
              if (idx === job.completedCount) {
                // Determine verdict/score randomly for demo
                const scoreVal = Math.random();
                return {
                  ...f,
                  status: 'completed' as const,
                  verdict: scoreVal > 0.4 ? ('fake' as const) : ('real' as const),
                  score: parseFloat(scoreVal.toFixed(2))
                };
              }
              if (idx === nextCompleted && idx < job.files.length) {
                return { ...f, status: 'processing' as const };
              }
              return f;
            });
            
            const nextStatus = nextCompleted >= job.totalFiles ? ('completed' as const) : ('processing' as const);

            return {
              ...job,
              completedCount: Math.min(job.totalFiles, nextCompleted),
              status: nextStatus,
              files: updatedFiles
            };
          }
          return job;
        })
      );
    }, 4500);

    return () => clearInterval(interval);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!urlsInput.trim()) return;

    const lines = urlsInput.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return;

    const newJobFiles: BatchFile[] = lines.map(line => ({
      filename: line.substring(line.lastIndexOf('/') + 1) || line,
      verdict: 'real',
      score: 0.0,
      status: 'pending'
    }));

    // Start first file as processing
    newJobFiles[0].status = 'processing';

    const newJob: BatchJob = {
      id: `BATCH-2026-${Math.floor(1000 + Math.random() * 9000)}`,
      priority,
      webhookUrl: webhookUrl || 'No Callback Registered',
      totalFiles: newJobFiles.length,
      completedCount: 0,
      status: 'processing',
      files: newJobFiles,
      timestamp: new Date().toISOString().replace('T', ' ').substring(0, 16)
    };

    setJobs(prev => [newJob, ...prev]);
    setExpandedJobId(newJob.id);
    setUrlsInput('');
    setWebhookUrl('');
  };

  const handlePause = (jobId: string) => {
    setJobs(prev => prev.map(j => {
      if (j.id === jobId) {
        return { ...j, status: j.status === 'paused' ? 'processing' : 'paused' };
      }
      return j;
    }));
  };

  const handleCancel = (jobId: string) => {
    setJobs(prev => prev.filter(j => j.id !== jobId));
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white glow-text">Batch Pipeline Panel</h1>
          <p className="text-slate-400 text-sm mt-1">Submit high-volume URLs, monitor asynchronous priority queues, and configure webhook delivery.</p>
        </div>
        <div className="glass px-4 py-2 rounded-lg flex items-center space-x-2 text-slate-300 font-mono text-xs">
          <Layers className="h-4 w-4 text-cyan-400" />
          <span>ROUTER: ACTIVE [T1/T2/T3]</span>
        </div>
      </div>

      {/* Grid: Submission Form on Left, Active Queue on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Submission Form (1 col) */}
        <div className="glass-card p-6 rounded-xl space-y-6 h-fit">
          <div>
            <h2 className="text-base font-bold text-white">New Batch Ingestion</h2>
            <p className="text-xs text-slate-400 mt-0.5">Submit files, paths, or URLs (one per line).</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Media URLs textarea */}
            <div className="space-y-1.5">
              <label htmlFor="batch-urls" className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Payload URLs / Paths</label>
              <textarea
                id="batch-urls"
                rows={5}
                placeholder="https://bucket.s3.amazonaws.com/media/leak_clip.mp4&#10;d:/shiva reddy project/assets/ceo_voice.wav&#10;https://newsagency.org/press_photo.jpg"
                value={urlsInput}
                onChange={(e) => setUrlsInput(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            {/* Priority Select */}
            <div className="space-y-1.5">
              <label htmlFor="batch-priority" className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Queue Priority</label>
              <select
                id="batch-priority"
                value={priority}
                onChange={(e) => setPriority(e.target.value as any)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
              >
                <option value="low">Low Priority (T1 Background evaluation)</option>
                <option value="medium">Medium Priority (Standard T1-T2 pipeline)</option>
                <option value="high">High Priority (Immediate T3 Full-Stack Audit)</option>
              </select>
            </div>

            {/* Webhook Callback */}
            <div className="space-y-1.5">
              <label htmlFor="batch-webhook" className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                <span>Webhook Notification URL</span>
                <span className="text-[10px] text-slate-500 font-normal">Optional</span>
              </label>
              <input
                id="batch-webhook"
                type="url"
                placeholder="https://api.enterprise.com/callback"
                value={webhookUrl}
                onChange={(e) => setWebhookUrl(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            {/* Submit */}
            <button
              id="btn-submit-batch"
              type="submit"
              className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center space-x-1 hover:brightness-110 transition-all shadow-md shadow-cyan-500/10 cursor-pointer"
            >
              <Send className="h-3.5 w-3.5" />
              <span>Queue Batch Inferences</span>
            </button>
          </form>
        </div>

        {/* Active Queue listing (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card p-6 rounded-xl">
            <h2 className="text-base font-bold text-white mb-6">Asynchronous Pipeline Logs</h2>

            <div className="space-y-6">
              {jobs.map(job => (
                <div key={job.id} className="border border-slate-900 rounded-lg bg-slate-950/20 overflow-hidden">
                  
                  {/* Job Header Row */}
                  <div className="p-4 border-b border-slate-900 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div className="flex items-center space-x-3">
                      <button 
                        onClick={() => setExpandedJobId(expandedJobId === job.id ? null : job.id)}
                        className="p-1 text-slate-500 hover:text-white"
                      >
                        <ChevronRight className={`h-4 w-4 transition-transform duration-200 ${expandedJobId === job.id ? 'transform rotate-90' : ''}`} />
                      </button>
                      <div>
                        <span className="font-mono text-xs text-cyan-400 font-bold">{job.id}</span>
                        <div className="flex items-center space-x-2 text-[10px] text-slate-500 font-mono mt-0.5">
                          <span>SUBMITTED: {job.timestamp}</span>
                          <span>&bull;</span>
                          <span className={`uppercase font-bold ${
                            job.priority === 'high' ? 'text-rose-400' : job.priority === 'medium' ? 'text-amber-400' : 'text-slate-400'
                          }`}>{job.priority} Priority</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-4">
                      {/* Progress text */}
                      <span className="font-mono text-xs text-slate-400">
                        {job.completedCount}/{job.totalFiles} files completed
                      </span>

                      {/* Status Badge */}
                      <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                        job.status === 'completed' 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                          : job.status === 'processing'
                          ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 animate-pulse'
                          : job.status === 'paused'
                          ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                          : 'bg-slate-500/10 text-slate-400'
                      }`}>
                        {job.status === 'completed' && <CheckCircle2 className="h-3 w-3 mr-1" />}
                        {job.status === 'processing' && <Clock className="h-3 w-3 mr-1" />}
                        <span>{job.status}</span>
                      </span>

                      {/* Pause / cancel triggers */}
                      {job.status !== 'completed' && (
                        <div className="flex items-center space-x-1.5">
                          <button
                            onClick={() => handlePause(job.id)}
                            className="p-1 hover:text-white text-slate-500"
                            title={job.status === 'paused' ? 'Resume' : 'Pause'}
                          >
                            {job.status === 'paused' ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
                          </button>
                          <button
                            onClick={() => handleCancel(job.id)}
                            className="p-1 hover:text-white text-slate-500"
                            title="Cancel job"
                          >
                            <XCircle className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="h-1 bg-slate-900 w-full">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        job.status === 'completed' ? 'bg-emerald-500' : 'bg-cyan-500'
                      }`}
                      style={{ width: `${(job.completedCount / job.totalFiles) * 100}%` }}
                    ></div>
                  </div>

                  {/* Expanded job files ledger */}
                  {expandedJobId === job.id && (
                    <div className="p-4 bg-slate-950/40 text-xs font-mono">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">
                        WEBHOOK CALLBACK: {job.webhookUrl}
                      </div>
                      
                      <div className="divide-y divide-slate-900 border border-slate-900 rounded-lg overflow-hidden max-h-56 overflow-y-auto bg-black/40">
                        {job.files.map((file, idx) => (
                          <div key={idx} className="flex justify-between items-center py-2 px-3 text-[11px] hover:bg-slate-900/10">
                            <span className="text-slate-300 truncate max-w-xs">{file.filename}</span>
                            
                            <div className="flex items-center space-x-3">
                              {file.status === 'completed' ? (
                                <>
                                  <span className={`font-semibold ${file.verdict === 'fake' ? 'text-rose-400' : 'text-emerald-400'}`}>
                                    {(file.score * 100).toFixed(0)}%
                                  </span>
                                  <span className={`inline-flex px-1 rounded-[3px] text-[9px] font-semibold uppercase ${
                                    file.verdict === 'fake' ? 'bg-rose-500/15 text-rose-400' : 'bg-emerald-500/15 text-emerald-400'
                                  }`}>
                                    {file.verdict === 'fake' ? 'AI Manipulated' : 'Safe'}
                                  </span>
                                </>
                              ) : (
                                <span className={`text-[10px] uppercase ${
                                  file.status === 'processing' ? 'text-cyan-400 animate-pulse' : 'text-slate-600'
                                }`}>
                                  {file.status}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
