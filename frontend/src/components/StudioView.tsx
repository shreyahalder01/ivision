import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipBack,
  SkipForward,
  Upload,
  Sparkles,
  Sliders,
  Eye,
  Crosshair,
  Maximize2,
  Check,
  Search,
  Zap,
  Gauge,
  AlertCircle,
  Clock,
  Activity,
  Layers,
  ChevronRight,
  RefreshCw,
} from 'lucide-react';
import {
  SystemCapabilities,
  SampleVideo,
  Job,
  ResultsOverlayPayload,
} from '../types';
import { CanvasOverlay } from './CanvasOverlay';
import { uploadVideo, createJob, cancelJob, resumeJob, getJobResults, getMediaUrl } from '../api';

interface StudioViewProps {
  systemCaps: SystemCapabilities | null;
  sampleVideos: SampleVideo[];
  currentJob: Job | null;
  setCurrentJob: (job: Job | null) => void;
  onJobComplete: (job: Job) => void;
}

export const StudioView: React.FC<StudioViewProps> = ({
  systemCaps,
  sampleVideos,
  currentJob,
  setCurrentJob,
  onJobComplete,
}) => {
  // Upload & Video Source state
  const [selectedVideoPath, setSelectedVideoPath] = useState<string>('');
  const [selectedVideoName, setSelectedVideoName] = useState<string>('');
  const [videoMetadata, setVideoMetadata] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadPercent, setUploadPercent] = useState(0);

  // AI Configuration state
  const [selectedModel, setSelectedModel] = useState<string>('auto');
  const [selectedTracker, setSelectedTracker] = useState<string>('auto');
  const [confidence, setConfidence] = useState<number>(0.30);
  const [iou, setIou] = useState<number>(0.45);
  const [frameStride, setFrameStride] = useState<number>(1);
  const [annotationStyle, setAnnotationStyle] = useState<string>('box_label');
  const [selectedClasses, setSelectedClasses] = useState<string[]>([
    'person',
    'car',
    'truck',
    'bus',
    'bicycle',
    'motorcycle',
  ]);
  const [classSearch, setClassSearch] = useState('');
  const [activeClassCategory, setActiveClassCategory] = useState<'featured' | 'vehicles' | 'people' | 'animals' | 'all'>('featured');

  // Video Playback & Overlay state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [resultsOverlay, setResultsOverlay] = useState<ResultsOverlayPayload | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // Overlay visual filters
  const [showTrails, setShowTrails] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [minConfidenceFilter, setMinConfidenceFilter] = useState(0.20);

  // Pre-load default sample video if none selected
  useEffect(() => {
    if (!selectedVideoPath && sampleVideos.length > 0) {
      const sample = sampleVideos.find((s) => s.filename.includes('traffic')) || sampleVideos[0];
      setSelectedVideoPath(sample.videoPath);
      setSelectedVideoName(sample.filename);
      setVideoMetadata(sample.metadata);
    }
  }, [sampleVideos, selectedVideoPath]);

  // Load results overlay whenever currentJob completes or has results
  useEffect(() => {
    if (currentJob && (currentJob.status === 'complete' || currentJob.status === 'analyzing')) {
      getJobResults(currentJob.id)
        .then((data) => setResultsOverlay(data))
        .catch(() => {});
    }
  }, [currentJob?.status, currentJob?.id]);

  // Handle File Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadPercent(0);
    try {
      const res = await uploadVideo(file, (pct) => setUploadPercent(pct));
      setSelectedVideoPath(res.videoPath);
      setSelectedVideoName(res.filename);
      setVideoMetadata(res.metadata);
    } catch (err: any) {
      alert(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  // Handle Sample Select
  const handleSelectSample = (sample: SampleVideo) => {
    setSelectedVideoPath(sample.videoPath);
    setSelectedVideoName(sample.filename);
    setVideoMetadata(sample.metadata);
    setResultsOverlay(null);
  };

  // Run AI Analysis
  const handleStartAnalysis = async () => {
    if (!selectedVideoPath) {
      alert('Please select or upload a video first.');
      return;
    }

    try {
      const job = await createJob({
        video_path: selectedVideoPath,
        original_name: selectedVideoName,
        classes: selectedClasses,
        confidence,
        iou,
        model: selectedModel,
        tracking_method: selectedTracker,
        annotation_style: annotationStyle,
        frame_stride: frameStride,
      });
      setCurrentJob(job);
      setResultsOverlay(null);
    } catch (err: any) {
      alert(err.message || 'Failed to start analysis');
    }
  };

  // Cancel / Resume Job
  const handleCancelJob = async () => {
    if (!currentJob) return;
    try {
      await cancelJob(currentJob.id);
    } catch (err: any) {
      console.error(err);
    }
  };

  const handleResumeJob = async () => {
    if (!currentJob) return;
    try {
      await resumeJob(currentJob.id);
    } catch (err: any) {
      console.error(err);
    }
  };

  // Video Controls
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const stepFrame = (forward: boolean) => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    setIsPlaying(false);
    const fps = videoMetadata?.fps || 30.0;
    const step = (1 / fps) * (forward ? 1 : -1);
    videoRef.current.currentTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + step));
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const changePlaybackRate = (rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  const jumpToTrackTime = (timeSec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timeSec;
      setCurrentTime(timeSec);
    }
  };

  // Class selection filtering
  const allCoco = systemCaps?.cocoClasses || {};
  const featured = systemCaps?.featuredClasses || [];
  const groups = systemCaps?.classGroups || {};

  const getFilteredClasses = () => {
    let list: string[] = [];
    if (activeClassCategory === 'featured') {
      list = featured;
    } else if (activeClassCategory === 'vehicles') {
      list = groups.vehicles || [];
    } else if (activeClassCategory === 'people') {
      list = groups.people || [];
    } else if (activeClassCategory === 'animals') {
      list = groups.animals || [];
    } else {
      list = Object.values(allCoco);
    }

    if (classSearch.trim()) {
      const q = classSearch.toLowerCase();
      return list.filter((c) => c.toLowerCase().includes(q));
    }
    return list;
  };

  const toggleClass = (name: string) => {
    if (selectedClasses.includes(name)) {
      setSelectedClasses(selectedClasses.filter((c) => c !== name));
    } else {
      setSelectedClasses([...selectedClasses, name]);
    }
  };

  const selectAllCategory = () => {
    const categoryClasses = getFilteredClasses();
    const union = Array.from(new Set([...selectedClasses, ...categoryClasses]));
    setSelectedClasses(union);
  };

  const clearCategory = () => {
    const categoryClasses = getFilteredClasses();
    setSelectedClasses(selectedClasses.filter((c) => !categoryClasses.includes(c)));
  };

  const isJobRunning = currentJob && (currentJob.status === 'extracting' || currentJob.status === 'analyzing' || currentJob.status === 'queued');
  const fpsNumber = videoMetadata?.fps || 30.0;
  const currentFrameNumber = Math.round(currentTime * fpsNumber);
  const totalFrameCount = videoMetadata?.frame_count || Math.round(duration * fpsNumber);

  return (
    <div className="space-y-6">
      {/* Top Split Workspace: Video & Player Canvas (Left 7 cols) + AI Config & HUD (Right 5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: Video Player & Canvas HUD (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col space-y-4">
          <div className="relative rounded-2xl overflow-hidden glass-panel border border-slate-800 shadow-2xl bg-carbon-950 flex flex-col">
            
            {/* Player Header Bar */}
            <div className="flex items-center justify-between px-4 py-2.5 bg-carbon-900/90 border-b border-slate-800/80 text-xs">
              <div className="flex items-center gap-2 font-mono text-slate-300 truncate">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                <span className="truncate font-semibold">{selectedVideoName || 'No footage loaded'}</span>
                {videoMetadata && (
                  <span className="text-slate-500 text-[11px]">
                    ({videoMetadata.width}×{videoMetadata.height} • {videoMetadata.fps?.toFixed(0)} FPS)
                  </span>
                )}
              </div>

              {/* Quick visual overlay toggles */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowTrails(!showTrails)}
                  title="Toggle Motion Trajectory Trails"
                  className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-all ${
                    showTrails ? 'bg-cyan-950/60 border-cyan-500/50 text-cyan-300' : 'bg-slate-900 border-slate-700 text-slate-500'
                  }`}
                >
                  Trails
                </button>
                <button
                  onClick={() => setShowLabels(!showLabels)}
                  title="Toggle Class/ID Badges"
                  className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-all ${
                    showLabels ? 'bg-cyan-950/60 border-cyan-500/50 text-cyan-300' : 'bg-slate-900 border-slate-700 text-slate-500'
                  }`}
                >
                  Labels
                </button>
                <button
                  onClick={() => setShowConfidence(!showConfidence)}
                  title="Toggle Confidence %"
                  className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-all ${
                    showConfidence ? 'bg-cyan-950/60 border-cyan-500/50 text-cyan-300' : 'bg-slate-900 border-slate-700 text-slate-500'
                  }`}
                >
                  Conf%
                </button>
              </div>
            </div>

            {/* Video + Synced Canvas Display */}
            <div className="relative aspect-video w-full bg-black flex items-center justify-center overflow-hidden select-none">
              {selectedVideoPath ? (
                <>
                  <video
                    ref={videoRef}
                    src={getMediaUrl(selectedVideoPath)}
                    className="w-full h-full object-contain pointer-events-none"
                    playsInline
                    preload="metadata"
                    onTimeUpdate={() => {
                      if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
                    }}
                    onLoadedMetadata={() => {
                      if (videoRef.current) {
                        setDuration(videoRef.current.duration);
                        if (!videoMetadata) {
                          setVideoMetadata({
                            width: videoRef.current.videoWidth,
                            height: videoRef.current.videoHeight,
                            duration: videoRef.current.duration,
                            fps: 30,
                            frame_count: Math.round(videoRef.current.duration * 30),
                          });
                        }
                      }
                    }}
                    onEnded={() => setIsPlaying(false)}
                  />
                  <CanvasOverlay
                    videoRef={videoRef}
                    results={resultsOverlay}
                    selectedTrackId={selectedTrackId}
                    onSelectTrack={(tid) => setSelectedTrackId(tid)}
                    annotationStyle={annotationStyle}
                    showTrails={showTrails}
                    showLabels={showLabels}
                    showConfidence={showConfidence}
                    minConfidenceFilter={minConfidenceFilter}
                  />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500 space-y-3">
                  <Upload className="w-10 h-10 text-slate-600 animate-bounce" />
                  <p className="text-sm">Upload a video or select a built-in sample to begin</p>
                </div>
              )}

              {/* Cyber Scanline effect */}
              <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent opacity-20 h-16 animate-scanline" />
            </div>

            {/* Playback Controls & Scrubber */}
            <div className="p-3 bg-carbon-900/90 border-t border-slate-800 space-y-2">
              {/* Scrubbing slider */}
              <div className="flex items-center gap-3">
                <span className="text-[11px] font-mono text-cyan-400 w-16 text-right">
                  {new Date(currentTime * 1000).toISOString().substr(14, 5)}
                </span>
                <input
                  type="range"
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={currentTime}
                  onChange={handleSeek}
                  className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer focus:outline-none"
                />
                <span className="text-[11px] font-mono text-slate-400 w-16">
                  {new Date((duration || 0) * 1000).toISOString().substr(14, 5)}
                </span>
              </div>

              {/* Bottom Buttons Bar */}
              <div className="flex items-center justify-between pt-1">
                {/* Frame Step & Play Controls */}
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => stepFrame(false)}
                    title="Step 1 Frame Back"
                    className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-all"
                  >
                    <SkipBack className="w-4 h-4" />
                  </button>

                  <button
                    onClick={togglePlay}
                    title={isPlaying ? 'Pause' : 'Play'}
                    className="p-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold shadow-lg shadow-cyan-500/25 transition-all hover:scale-105 active:scale-95"
                  >
                    {isPlaying ? <Pause className="w-4 h-4 fill-black" /> : <Play className="w-4 h-4 fill-black ml-0.5" />}
                  </button>

                  <button
                    onClick={() => stepFrame(true)}
                    title="Step 1 Frame Forward"
                    className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-all"
                  >
                    <SkipForward className="w-4 h-4" />
                  </button>

                  {/* Frame telemetry */}
                  <span className="ml-2 font-mono text-[11px] text-slate-400">
                    Frame <span className="text-cyan-400 font-bold">{currentFrameNumber}</span>
                    {totalFrameCount > 0 && <span> / {totalFrameCount}</span>}
                  </span>
                </div>

                {/* Speed selector & Confidence filter */}
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1 bg-slate-950/80 px-2 py-1 rounded-lg border border-slate-800 text-[11px] font-mono">
                    <span className="text-slate-400">Rate:</span>
                    {[0.5, 1, 2].map((r) => (
                      <button
                        key={r}
                        onClick={() => changePlaybackRate(r)}
                        className={`px-1.5 py-0.5 rounded ${playbackRate === r ? 'bg-cyan-500 text-black font-bold' : 'text-slate-400 hover:text-white'}`}
                      >
                        {r}x
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

          </div>

          {/* Built-in Sample Selector Row */}
          <div className="p-4 rounded-xl glass-panel border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>Fast-Start Test Footage:</span>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {sampleVideos.map((sample) => (
                <button
                  key={sample.name}
                  onClick={() => handleSelectSample(sample)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all ${
                    selectedVideoName === sample.filename
                      ? 'bg-cyan-950/80 border-cyan-500/60 text-cyan-300 shadow-sm shadow-cyan-500/20'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  {sample.name} ({sample.metadata?.duration?.toFixed(0)}s)
                </button>
              ))}
              
              {/* Custom Upload Button */}
              <label className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer transition-all">
                <Upload className="w-3.5 h-3.5 text-cyan-400" />
                <span>Upload Video</span>
                <input
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-matroska,video/webm"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          </div>

          {/* Upload Progress Alert if uploading */}
          {isUploading && (
            <div className="p-3 rounded-xl bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 text-xs flex items-center justify-between">
              <span>Uploading footage & extracting sprites...</span>
              <span className="font-mono font-bold">{uploadPercent}%</span>
            </div>
          )}

          {/* Active Track Inspector Cards if results available */}
          {resultsOverlay && resultsOverlay.tracks && resultsOverlay.tracks.length > 0 && (
            <div className="p-4 rounded-xl glass-panel border border-slate-800 space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold uppercase tracking-wider text-slate-300 font-mono flex items-center gap-1.5">
                  <Crosshair className="w-4 h-4 text-cyan-400" />
                  Identified Objects ({resultsOverlay.tracks.length})
                </span>
                <span className="text-slate-500 text-[11px]">Click object to highlight trajectory</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-h-48 overflow-y-auto pr-1">
                {resultsOverlay.tracks.map((trk) => {
                  const isSel = selectedTrackId === trk.trackId;
                  return (
                    <button
                      key={trk.trackId}
                      onClick={() => {
                        setSelectedTrackId(isSel ? null : trk.trackId);
                        jumpToTrackTime(trk.firstSeen);
                      }}
                      className={`p-2 rounded-lg text-left border transition-all ${
                        isSel
                          ? 'bg-cyan-950/80 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/20'
                          : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between text-[11px] font-mono font-bold">
                        <span>ID #{trk.trackId}</span>
                        <span className="text-cyan-400">{(trk.avgConfidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="text-xs font-semibold text-white capitalize mt-0.5 truncate">
                        {trk.className}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono mt-1">
                        {trk.duration.toFixed(1)}s • {trk.detectionCount} dets
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right: AI Configuration & Live Analysis HUD (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col space-y-4">
          
          {/* Live Progress HUD if Job Active */}
          {currentJob && isJobRunning && (
            <div className="p-5 rounded-2xl glass-panel border border-cyan-500/40 bg-gradient-to-b from-cyan-950/40 to-carbon-950 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-cyan-400 animate-ping" />
                  <span className="font-mono text-xs font-bold text-cyan-300 uppercase tracking-wider">
                    {currentJob.stage || 'Inference in progress'}
                  </span>
                </div>
                <span className="font-mono text-sm font-extrabold text-cyan-400">
                  {((currentJob.progress || 0) * 100).toFixed(1)}%
                </span>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${Math.max(2, (currentJob.progress || 0) * 100)}%` }}
                />
              </div>

              {/* Telemetry Numbers */}
              <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">FRAMES</div>
                  <div className="text-white font-bold mt-0.5">
                    {currentJob.processed_frames} / {currentJob.total_frames}
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">LIVE SPEED</div>
                  <div className="text-emerald-400 font-bold mt-0.5">
                    {currentJob.processing_fps ? `${currentJob.processing_fps.toFixed(1)} FPS` : '--'}
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-800">
                  <div className="text-slate-400 text-[10px]">DEVICE</div>
                  <div className="text-cyan-300 font-bold mt-0.5 truncate">
                    {currentJob.device || 'cuda:0'}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-end gap-2 pt-1">
                <button
                  onClick={handleCancelJob}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-600/40 transition-all"
                >
                  Cancel Run
                </button>
              </div>
            </div>
          )}

          {/* Analysis Settings Card */}
          <div className="p-5 rounded-2xl glass-panel border border-slate-800 space-y-5 bg-carbon-900/50">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <h3 className="font-bold text-sm text-slate-100 uppercase tracking-wider font-mono">
                  Detection & Tracking Parameters
                </h3>
              </div>
              <span className="text-[11px] font-mono text-cyan-400">YOLO11 Architecture</span>
            </div>

            {/* Model Family Selector */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                <span>Model Checkpoint</span>
                <span className="text-[11px] text-slate-500 font-normal">Auto optimizes for device</span>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setSelectedModel('auto')}
                  className={`p-2.5 rounded-xl text-left border transition-all ${
                    selectedModel === 'auto'
                      ? 'bg-cyan-950/80 border-cyan-400 text-cyan-200'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="font-bold text-xs font-mono">Auto (Adaptive)</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Device calibrated</div>
                </button>

                {systemCaps?.models?.map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setSelectedModel(m.key)}
                    className={`p-2.5 rounded-xl text-left border transition-all ${
                      selectedModel === m.key
                        ? 'bg-cyan-950/80 border-cyan-400 text-cyan-200'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-bold text-xs font-mono">{m.label}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      {m.speed} • {m.paramsM}M params
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Tracking Algorithm Mode */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300">
                Multi-Object Tracking Algorithm
              </label>
              <div className="grid grid-cols-2 gap-2">
                {systemCaps?.trackingMethods?.map((tm) => (
                  <button
                    key={tm.key}
                    onClick={() => setSelectedTracker(tm.key)}
                    className={`p-2 rounded-xl text-left border text-xs font-mono transition-all ${
                      selectedTracker === tm.key
                        ? 'bg-violet-950/80 border-violet-400 text-violet-200'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-bold">{tm.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders: Confidence Threshold & IoU */}
            <div className="space-y-4 pt-1">
              {/* Confidence */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-slate-300">Confidence Threshold</span>
                  <span className="font-mono text-cyan-400">{(confidence * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* IoU Overlap */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-slate-300">IoU Association Threshold</span>
                  <span className="font-mono text-violet-400">{(iou * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min={0.10}
                  max={0.90}
                  step={0.05}
                  value={iou}
                  onChange={(e) => setIou(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              {/* Frame Stride & Annotation Style */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Frame Stride</label>
                  <select
                    value={frameStride}
                    onChange={(e) => setFrameStride(parseInt(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value={1}>1 (Full 30 FPS)</option>
                    <option value={2}>2 (Every 2nd frame - 15 FPS)</option>
                    <option value={3}>3 (Every 3rd frame - 10 FPS)</option>
                    <option value={5}>5 (Fast 6 FPS)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300">Visual Style</label>
                  <select
                    value={annotationStyle}
                    onChange={(e) => setAnnotationStyle(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="box_label">Standard Box & Label</option>
                    <option value="corner_brackets">Corner Brackets</option>
                    <option value="cyber_hud">Cyber HUD</option>
                    <option value="minimal_dot">Minimal Center Dot</option>
                    <option value="mask">Alpha Shaded Box</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Target Classes Selector */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-300">
                  Target Classes ({selectedClasses.length} active)
                </label>
                <div className="flex items-center gap-2 text-[11px]">
                  <button
                    onClick={selectAllCategory}
                    className="text-cyan-400 hover:underline"
                  >
                    Select All
                  </button>
                  <span className="text-slate-600">•</span>
                  <button
                    onClick={clearCategory}
                    className="text-slate-400 hover:underline"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Category tabs */}
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px]">
                {(['featured', 'vehicles', 'people', 'animals', 'all'] as const).map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setActiveClassCategory(cat)}
                    className={`flex-1 py-1 rounded capitalize font-medium transition-all ${
                      activeClassCategory === cat
                        ? 'bg-slate-800 text-white font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Class chips grid */}
              <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto p-1 bg-slate-950/60 rounded-xl border border-slate-800">
                {getFilteredClasses().map((cls) => {
                  const isChecked = selectedClasses.includes(cls);
                  return (
                    <button
                      key={cls}
                      onClick={() => toggleClass(cls)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                        isChecked
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                          : 'bg-slate-900/60 text-slate-400 border border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <span>{cls}</span>
                      {isChecked && <Check className="w-3 h-3 text-cyan-400" />}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Launch CTA */}
            <button
              onClick={handleStartAnalysis}
              disabled={isJobRunning || !selectedVideoPath}
              className="w-full py-3.5 px-4 rounded-xl font-extrabold text-sm uppercase tracking-wider bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 text-black shadow-xl shadow-cyan-500/20 transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Zap className="w-4 h-4 fill-black" />
              <span>{isJobRunning ? 'Analysis in Progress...' : 'Run Real-Time AI Analysis'}</span>
            </button>
          </div>

        </div>

      </div>
    </div>
  );
};
