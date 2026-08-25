import React from 'react';
import {
  X,
  Cpu,
  CheckCircle2,
  AlertTriangle,
  HardDrive,
  Activity,
  Layers,
  Zap,
} from 'lucide-react';
import { SystemCapabilities } from '../types';

interface SystemModalProps {
  isOpen: boolean;
  onClose: () => void;
  systemCaps: SystemCapabilities | null;
}

export const SystemModal: React.FC<SystemModalProps> = ({
  isOpen,
  onClose,
  systemCaps,
}) => {
  if (!isOpen || !systemCaps) return null;

  const ai = systemCaps.ai;
  const ff = systemCaps.ffmpeg;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="relative w-full max-w-2xl rounded-2xl glass-panel border border-slate-700 bg-carbon-950 p-6 space-y-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-mono">
                Hardware Acceleration & Environment Diagnostics
              </h3>
              <p className="text-xs text-slate-400">VisionTrack AI Core v{systemCaps.version}</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Diagnostic Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* AI Inference Runtime */}
          <div className="p-4 rounded-xl bg-carbon-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-slate-300">AI INFERENCE ENGINE</span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                ai.aiAvailable ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {ai.aiAvailable ? 'READY' : 'OFFLINE'}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono text-slate-400">
              <div className="flex justify-between">
                <span>Device:</span>
                <span className="text-cyan-400 font-semibold">{ai.device} ({ai.deviceName})</span>
              </div>
              <div className="flex justify-between">
                <span>CUDA Support:</span>
                <span className={ai.cudaAvailable ? 'text-emerald-400' : 'text-slate-500'}>
                  {ai.cudaAvailable ? `CUDA ${ai.cudaVersion || 'Active'}` : 'Disabled'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>GPU VRAM:</span>
                <span className="text-slate-200">{ai.vramTotalGb ? `${ai.vramTotalGb} GB` : 'System RAM'}</span>
              </div>
              <div className="flex justify-between">
                <span>PyTorch:</span>
                <span className="text-slate-300">v{ai.torchVersion || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span>Ultralytics:</span>
                <span className="text-slate-300">v{ai.ultralyticsVersion || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Media & Video Codec Engine */}
          <div className="p-4 rounded-xl bg-carbon-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-slate-300">FFMPEG MEDIA CODEC</span>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                ff.ready ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {ff.ready ? 'ACTIVE' : 'DEGRADED'}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono text-slate-400">
              <div className="flex justify-between">
                <span>FFmpeg Binary:</span>
                <span className="text-slate-200 truncate max-w-[160px]" title={ff.ffmpeg.path || ''}>
                  {ff.ffmpeg.source || 'Detected'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>HW Encoder:</span>
                <span className="text-emerald-400 font-semibold">{ff.preferredEncoder}</span>
              </div>
              <div className="flex justify-between">
                <span>Available Encoders:</span>
                <span className="text-slate-300">{ff.hardwareEncoders?.join(', ') || 'libx264'}</span>
              </div>
              <div className="flex justify-between">
                <span>FFprobe:</span>
                <span className="text-slate-300">{ff.ffprobe.available ? 'Ready' : 'Missing'}</span>
              </div>
            </div>
          </div>

        </div>

        {/* YOLO11 Model Specs Table */}
        <div className="space-y-2">
          <div className="text-xs font-mono font-bold text-slate-300">SUPPORTED YOLO11 CHECKPOINTS</div>
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900 text-slate-400 text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">CHECKPOINT</th>
                  <th className="py-2 px-3">SPEED</th>
                  <th className="py-2 px-3">ACCURACY</th>
                  <th className="py-2 px-3">PARAMETERS</th>
                  <th className="py-2 px-3">MIN VRAM</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {systemCaps.models?.map((m) => (
                  <tr key={m.key} className="hover:bg-slate-900/50">
                    <td className="py-2 px-3 font-bold text-cyan-400">{m.label}</td>
                    <td className="py-2 px-3 text-slate-300">{m.speed}</td>
                    <td className="py-2 px-3 text-emerald-400">{m.accuracy}</td>
                    <td className="py-2 px-3 text-slate-400">{m.paramsM}M ({m.sizeMb} MB)</td>
                    <td className="py-2 px-3 text-slate-400">{m.minVramGb} GB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Close Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold font-mono transition-colors"
          >
            Close Diagnostics
          </button>
        </div>

      </div>
    </div>
  );
};
