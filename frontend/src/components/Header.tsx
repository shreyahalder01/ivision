import React from 'react';
import {
  Activity,
  Cpu,
  Layers,
  BarChart3,
  DownloadCloud,
  History,
  Info,
  Sparkles,
  Zap,
} from 'lucide-react';
import { SystemCapabilities } from '../types';

interface HeaderProps {
  activeTab: 'studio' | 'analytics' | 'exports' | 'history';
  onSelectTab: (tab: 'studio' | 'analytics' | 'exports' | 'history') => void;
  systemCaps: SystemCapabilities | null;
  onOpenSystemModal: () => void;
  onLoadSample: () => void;
  isProcessing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  systemCaps,
  onOpenSystemModal,
  onLoadSample,
  isProcessing,
}) => {
  const isGpu = systemCaps?.ai?.gpuAccelerated;
  const deviceName = systemCaps?.ai?.deviceName || 'Detecting...';

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-carbon-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-slate-900 to-violet-500/20 border border-cyan-500/40 glow-cyan">
            <Activity className="w-5 h-5 text-cyan-400 animate-pulse" />
            <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-carbon-950 animate-ping" />
            <div className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-carbon-950" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-200 to-indigo-300 text-lg font-mono">
                VISIONTRACK<span className="text-cyan-400">.AI</span>
              </span>
              <span className="text-[10px] font-mono font-semibold uppercase px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                v1.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              YOLO11 Persistent Multi-Object Tracking Engine
            </p>
          </div>
        </div>

        {/* Center Tabs */}
        <nav className="hidden md:flex items-center p-1 rounded-xl bg-carbon-900/90 border border-slate-800">
          <button
            onClick={() => onSelectTab('studio')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'studio'
                ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Studio Workspace
          </button>

          <button
            onClick={() => onSelectTab('analytics')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'analytics'
                ? 'bg-gradient-to-r from-violet-500/20 to-purple-500/20 text-violet-300 border border-violet-500/40 shadow-sm shadow-violet-500/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Analytics
          </button>

          <button
            onClick={() => onSelectTab('exports')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'exports'
                ? 'bg-gradient-to-r from-emerald-500/20 to-teal-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <DownloadCloud className="w-3.5 h-3.5" />
            Export Hub
          </button>

          <button
            onClick={() => onSelectTab('history')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'history'
                ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 border border-amber-500/40 shadow-sm shadow-amber-500/10'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            Job History
          </button>
        </nav>

        {/* Right Hardware Telemetry & Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Quick 1-Click Sample Button */}
          <button
            onClick={onLoadSample}
            disabled={isProcessing}
            title="Load built-in sample traffic video"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-600/20 via-cyan-500/30 to-blue-600/20 hover:from-cyan-600/30 hover:to-blue-600/30 text-cyan-300 border border-cyan-500/50 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-spin-slow" />
            <span className="hidden sm:inline">Load Sample</span>
          </button>

          {/* Hardware Acceleration Pill */}
          <button
            onClick={onOpenSystemModal}
            className={`flex items-center gap-2 px-2.5 py-1 rounded-lg border text-xs font-mono transition-all ${
              isGpu
                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300 hover:bg-emerald-900/40'
                : 'bg-amber-950/40 border-amber-500/40 text-amber-300 hover:bg-amber-900/40'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span className="hidden sm:inline max-w-[130px] truncate">{deviceName}</span>
            <span className="flex h-2 w-2 relative">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isGpu ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isGpu ? 'bg-emerald-500' : 'bg-amber-500'}`} />
            </span>
          </button>
        </div>

      </div>

      {/* Mobile Sub-Navigation Bar */}
      <div className="md:hidden flex items-center justify-around px-2 py-1.5 border-t border-slate-800 bg-carbon-900/90 text-xs">
        <button
          onClick={() => onSelectTab('studio')}
          className={`flex items-center gap-1 px-2.5 py-1 rounded ${activeTab === 'studio' ? 'text-cyan-400 font-bold bg-cyan-950/60' : 'text-slate-400'}`}
        >
          <Layers className="w-3 h-3" /> Studio
        </button>
        <button
          onClick={() => onSelectTab('analytics')}
          className={`flex items-center gap-1 px-2.5 py-1 rounded ${activeTab === 'analytics' ? 'text-violet-400 font-bold bg-violet-950/60' : 'text-slate-400'}`}
        >
          <BarChart3 className="w-3 h-3" /> Analytics
        </button>
        <button
          onClick={() => onSelectTab('exports')}
          className={`flex items-center gap-1 px-2.5 py-1 rounded ${activeTab === 'exports' ? 'text-emerald-400 font-bold bg-emerald-950/60' : 'text-slate-400'}`}
        >
          <DownloadCloud className="w-3 h-3" /> Export
        </button>
        <button
          onClick={() => onSelectTab('history')}
          className={`flex items-center gap-1 px-2.5 py-1 rounded ${activeTab === 'history' ? 'text-amber-400 font-bold bg-amber-950/60' : 'text-slate-400'}`}
        >
          <History className="w-3 h-3" /> History
        </button>
      </div>
    </header>
  );
};
