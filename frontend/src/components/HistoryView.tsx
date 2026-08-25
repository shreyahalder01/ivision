import React, { useState, useEffect } from 'react';
import {
  History,
  Trash2,
  Play,
  CheckCircle2,
  AlertCircle,
  Clock,
  Crosshair,
  Layers,
  ArrowRight,
  RefreshCw,
} from 'lucide-react';
import { Job } from '../types';
import { listJobs, deleteJob, getMediaUrl } from '../api';

interface HistoryViewProps {
  onSelectJob: (job: Job) => void;
  onSwitchToStudio: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  onSelectJob,
  onSwitchToStudio,
}) => {
  const [jobsList, setJobsList] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchJobs = async () => {
    try {
      setIsLoading(true);
      const res = await listJobs(100, 0);
      setJobsList(res.items);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  const handleDeleteJob = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (!confirm(`Are you sure you want to delete job #${jobId}?`)) return;
    try {
      await deleteJob(jobId);
      await fetchJobs();
    } catch (err: any) {
      alert(err.message || 'Failed to delete job');
    }
  };

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="p-6 rounded-2xl glass-panel border border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-amber-950/60 border border-amber-500/30 text-amber-400">
            <History className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-100 font-mono">Analysis History</h2>
            <p className="text-xs text-slate-400">Review past detection & tracking sessions</p>
          </div>
        </div>

        <button
          onClick={fetchJobs}
          disabled={isLoading}
          className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Grid of Past Jobs */}
      {jobsList.length === 0 ? (
        <div className="p-12 text-center glass-panel rounded-2xl border border-slate-800 space-y-4">
          <Clock className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-lg font-bold text-slate-200">No Prior Analysis Runs</h3>
          <p className="text-sm text-slate-400 max-w-md mx-auto">
            Jobs you run will be saved in SQLite with all detections, track IDs, and telemetry intact.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {jobsList.map((job) => {
            const hasResults = job.results != null;
            return (
              <div
                key={job.id}
                onClick={() => {
                  onSelectJob(job);
                  onSwitchToStudio();
                }}
                className="p-5 rounded-2xl glass-panel border border-slate-800 hover:border-cyan-500/40 bg-carbon-900/60 transition-all cursor-pointer group flex flex-col justify-between space-y-4"
              >
                {/* Poster / Thumbnail or Placeholder */}
                <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 border border-slate-800">
                  {job.poster_path ? (
                    <img
                      src={getMediaUrl(job.poster_path)}
                      alt={job.original_name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-700">
                      <Play className="w-8 h-8 opacity-40" />
                    </div>
                  )}

                  {/* Status Badge */}
                  <div className="absolute top-2 left-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                      job.status === 'complete'
                        ? 'bg-emerald-950/90 text-emerald-400 border border-emerald-700/60'
                        : job.status === 'analyzing'
                        ? 'bg-cyan-950/90 text-cyan-400 border border-cyan-700/60 animate-pulse'
                        : 'bg-slate-900/90 text-slate-400 border border-slate-700/60'
                    }`}>
                      {job.status}
                    </span>
                  </div>

                  {/* Time */}
                  <div className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-black/80 text-[10px] font-mono text-slate-300">
                    {new Date(job.created_at * 1000).toLocaleDateString()}
                  </div>
                </div>

                {/* Job Metadata */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-slate-100 truncate group-hover:text-cyan-300 transition-colors">
                      {job.original_name}
                    </h3>
                  </div>

                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-400">
                    <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                      <div className="text-[10px] text-slate-500">OBJECTS</div>
                      <div className="text-cyan-400 font-bold mt-0.5">
                        {job.results?.uniqueObjects ?? '--'}
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
                      <div className="text-[10px] text-slate-500">DETECTIONS</div>
                      <div className="text-violet-400 font-bold mt-0.5">
                        {job.results?.totalDetections ?? '--'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Bar */}
                <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-xs font-mono">
                  <span className="text-cyan-400 group-hover:underline flex items-center gap-1">
                    <span>Load Session</span>
                    <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                  </span>

                  <button
                    onClick={(e) => handleDeleteJob(e, job.id)}
                    title="Delete Job"
                    className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/40 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
};
