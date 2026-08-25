import React, { useState, useEffect } from 'react';
import {
  DownloadCloud,
  FileVideo,
  FileText,
  FileSpreadsheet,
  Layers,
  Database,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Download,
  Loader2,
  Sparkles,
  Zap,
} from 'lucide-react';
import { Job, ExportRecord } from '../types';
import { createExport, listJobExports, getExportDownloadUrl } from '../api';

interface ExportViewProps {
  currentJob: Job | null;
  onSwitchToStudio: () => void;
}

export const ExportView: React.FC<ExportViewProps> = ({
  currentJob,
  onSwitchToStudio,
}) => {
  const [exportsList, setExportsList] = useState<ExportRecord[]>([]);
  const [isLoadingExports, setIsLoadingExports] = useState(false);
  const [activeTabFormat, setActiveTabFormat] = useState<'mp4' | 'json' | 'csv' | 'coco' | 'yolo'>('mp4');

  // Video Export options
  const [videoResolution, setVideoResolution] = useState<'source' | '1080' | '720' | '540'>('source');
  const [videoFps, setVideoFps] = useState<number>(30);
  const [videoStyle, setVideoStyle] = useState<string>('box_label');
  const [burnTelemetry, setBurnTelemetry] = useState(true);

  // General export state
  const [isExporting, setIsExporting] = useState(false);

  const fetchExports = async () => {
    if (!currentJob) return;
    try {
      setIsLoadingExports(true);
      const list = await listJobExports(currentJob.id);
      setExportsList(list);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoadingExports(false);
    }
  };

  useEffect(() => {
    fetchExports();
    const interval = setInterval(fetchExports, 2500);
    return () => clearInterval(interval);
  }, [currentJob?.id]);

  if (!currentJob) {
    return (
      <div className="p-12 text-center glass-panel rounded-2xl border border-slate-800 space-y-4">
        <DownloadCloud className="w-12 h-12 text-slate-600 mx-auto animate-bounce" />
        <h3 className="text-lg font-bold text-slate-200">No Job Selected For Export</h3>
        <p className="text-sm text-slate-400 max-w-md mx-auto">
          Run or select an analysis job first in the Studio Workspace to export videos, bounding boxes, and datasets.
        </p>
        <button
          onClick={onSwitchToStudio}
          className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs uppercase tracking-wider transition-all"
        >
          Go to Studio Workspace
        </button>
      </div>
    );
  }

  const handleTriggerExport = async (format: string) => {
    if (!currentJob) return;
    setIsExporting(true);
    try {
      let options: any = {};
      if (format === 'mp4') {
        options = {
          width: videoResolution === '1080' ? 1920 : videoResolution === '720' ? 1280 : videoResolution === '540' ? 960 : null,
          fps: videoFps,
          style: videoStyle,
          timecode: burnTelemetry,
        };
      }

      await createExport(currentJob.id, format, options);
      await fetchExports();
    } catch (err: any) {
      alert(err.message || 'Failed to trigger export');
    } finally {
      setIsExporting(false);
    }
  };

  const formats = [
    {
      id: 'mp4',
      name: 'Annotated MP4 Video',
      icon: FileVideo,
      badge: 'H.264 / NVENC',
      desc: 'High-definition video with persistent AI bounding boxes, IDs, class labels, and telemetry burned directly into frames.',
      color: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/40 text-cyan-300',
    },
    {
      id: 'json',
      name: 'Structured JSON Payload',
      icon: FileText,
      badge: 'Full Schema v1',
      desc: 'Compact frame-by-frame 0..1 bounding box coordinates, track trajectories, velocity vectors, and complete metadata.',
      color: 'from-violet-500/20 to-purple-500/20 border-violet-500/40 text-violet-300',
    },
    {
      id: 'csv',
      name: 'Tabular CSV Dataset',
      icon: FileSpreadsheet,
      badge: 'Dataframe Ready',
      desc: 'Flat rectangular schema with (frame, track_id, class_name, confidence, x, y, width, height) ready for Pandas or Excel.',
      color: 'from-emerald-500/20 to-teal-500/20 border-emerald-500/40 text-emerald-300',
    },
    {
      id: 'coco',
      name: 'COCO Benchmark Format',
      icon: Database,
      badge: 'Standard Benchmark',
      desc: 'Standard MS-COCO JSON annotations containing image metadata, category maps, and bounding box entries.',
      color: 'from-amber-500/20 to-orange-500/20 border-amber-500/40 text-amber-300',
    },
    {
      id: 'yolo',
      name: 'YOLO Training Archive',
      icon: Layers,
      badge: 'ZIP Dataset',
      desc: 'ZIP archive with normalized bounding box txt files for each frame along with classes.txt and data.yaml configuration.',
      color: 'from-rose-500/20 to-pink-500/20 border-rose-500/40 text-rose-300',
    },
  ];

  return (
    <div className="space-y-6">
      
      {/* Top Header */}
      <div className="p-6 rounded-2xl glass-panel border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <DownloadCloud className="w-5 h-5 text-emerald-400" />
            <h2 className="text-lg font-bold text-slate-100 font-mono">
              Export & Dataset Hub
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Target Job: <span className="text-cyan-400 font-mono font-bold">#{currentJob.id}</span> ({currentJob.original_name})
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300">
            Detections: <span className="text-cyan-400 font-bold">{currentJob.results?.totalDetections || 0}</span>
          </div>
          <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300">
            Tracks: <span className="text-violet-400 font-bold">{currentJob.results?.uniqueObjects || 0}</span>
          </div>
        </div>
      </div>

      {/* Grid of 5 Export Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {formats.map((fmt) => {
          const Icon = fmt.icon;
          return (
            <div
              key={fmt.id}
              className={`p-5 rounded-2xl glass-panel border bg-gradient-to-b flex flex-col justify-between space-y-4 hover:scale-[1.01] transition-all ${fmt.color}`}
            >
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800">
                    {fmt.badge}
                  </span>
                </div>

                <h3 className="font-bold text-sm text-white">{fmt.name}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{fmt.desc}</p>
              </div>

              {/* Format Specific Options */}
              {fmt.id === 'mp4' && (
                <div className="space-y-2 pt-2 border-t border-slate-800 text-xs font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Resolution:</span>
                    <select
                      value={videoResolution}
                      onChange={(e) => setVideoResolution(e.target.value as any)}
                      className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-[11px] text-slate-200"
                    >
                      <option value="source">Source Native</option>
                      <option value="1080">1080p Full HD</option>
                      <option value="720">720p HD</option>
                      <option value="540">540p Fast</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Overlay Style:</span>
                    <select
                      value={videoStyle}
                      onChange={(e) => setVideoStyle(e.target.value)}
                      className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-[11px] text-slate-200"
                    >
                      <option value="box_label">Standard Box</option>
                      <option value="corner_brackets">Brackets</option>
                      <option value="cyber_hud">Cyber HUD</option>
                    </select>
                  </div>
                </div>
              )}

              <button
                onClick={() => handleTriggerExport(fmt.id)}
                disabled={isExporting}
                className="w-full py-2.5 px-3 rounded-xl font-bold text-xs uppercase tracking-wider bg-slate-900 hover:bg-slate-800 text-white border border-slate-700 hover:border-slate-500 transition-all flex items-center justify-center gap-2"
              >
                {isExporting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
                <span>Generate {fmt.id.toUpperCase()}</span>
              </button>
            </div>
          );
        })}
      </div>

      {/* Generated Artifacts Table */}
      <div className="p-6 rounded-2xl glass-panel border border-slate-800 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Download className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider font-mono">
              Generated Export Artifacts ({exportsList.length})
            </h3>
          </div>
          {isLoadingExports && <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />}
        </div>

        {exportsList.length === 0 ? (
          <div className="py-8 text-center text-xs font-mono text-slate-500">
            No exports have been generated for this job yet. Click a format card above to produce an artifact.
          </div>
        ) : (
          <div className="divide-y divide-slate-800/60">
            {exportsList.map((exp) => (
              <div key={exp.id} className="py-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 font-mono text-xs">
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center font-bold text-cyan-400 uppercase">
                    {exp.fmt}
                  </span>
                  <div>
                    <div className="font-semibold text-slate-200">
                      Export #{exp.id} ({exp.kind})
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Created: {new Date(exp.created_at * 1000).toLocaleTimeString()}
                      {exp.size_bytes ? ` • ${(exp.size_bytes / 1024).toFixed(1)} KB` : ''}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
                  {/* Status Badge */}
                  {exp.status === 'complete' && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-800/40">
                      <CheckCircle2 className="w-3 h-3" /> Ready
                    </span>
                  )}
                  {exp.status === 'running' && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-md border border-cyan-800/40">
                      <Loader2 className="w-3 h-3 animate-spin" /> {((exp.progress || 0) * 100).toFixed(0)}%
                    </span>
                  )}
                  {exp.status === 'failed' && (
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-rose-400 bg-rose-950/60 px-2.5 py-1 rounded-md border border-rose-800/40">
                      <AlertTriangle className="w-3 h-3" /> Failed
                    </span>
                  )}

                  {/* Download Button */}
                  {exp.status === 'complete' && (
                    <a
                      href={getExportDownloadUrl(exp.id)}
                      download
                      className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs transition-all flex items-center gap-1.5 shadow-sm shadow-cyan-500/20"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download File</span>
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};
