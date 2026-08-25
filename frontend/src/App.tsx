import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { StudioView } from './components/StudioView';
import { AnalyticsView } from './components/AnalyticsView';
import { ExportView } from './components/ExportView';
import { HistoryView } from './components/HistoryView';
import { SystemModal } from './components/SystemModal';
import {
  SystemCapabilities,
  SampleVideo,
  Job,
  ResultsOverlayPayload,
} from './types';
import {
  fetchCapabilities,
  fetchSampleVideos,
  getJob,
  getJobResults,
  connectJobWebSocket,
} from './api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'studio' | 'analytics' | 'exports' | 'history'>('studio');
  const [systemCaps, setSystemCaps] = useState<SystemCapabilities | null>(null);
  const [sampleVideos, setSampleVideos] = useState<SampleVideo[]>([]);
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [resultsOverlay, setResultsOverlay] = useState<ResultsOverlayPayload | null>(null);
  const [isSystemModalOpen, setIsSystemModalOpen] = useState(false);

  // Load initial system data
  useEffect(() => {
    fetchCapabilities()
      .then((caps) => setSystemCaps(caps))
      .catch((err) => console.error('Capabilities error', err));

    fetchSampleVideos()
      .then((samples) => setSampleVideos(samples))
      .catch((err) => console.error('Sample videos error', err));
  }, []);

  // Subscribe to WebSocket events when currentJob is active
  useEffect(() => {
    if (!currentJob) return;

    // If job is finished or ready, load overlay
    if (currentJob.status === 'complete') {
      getJobResults(currentJob.id)
        .then((res) => setResultsOverlay(res))
        .catch(() => {});
      return;
    }

    const unsubscribe = connectJobWebSocket(currentJob.id, (event) => {
      if (event.type === 'progress') {
        setCurrentJob((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            progress: event.progress,
            processed_frames: event.processed,
            total_frames: event.total,
            processing_fps: event.rate,
            liveProgress: event,
          };
        });
      } else if (event.type === 'status') {
        setCurrentJob((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            status: event.status,
            stage: event.stage,
          };
        });
      } else if (event.type === 'job_complete' || event.status === 'complete') {
        getJob(currentJob.id)
          .then((updated) => {
            setCurrentJob(updated);
            getJobResults(currentJob.id).then((res) => setResultsOverlay(res));
          })
          .catch(() => {});
      }
    });

    return () => unsubscribe();
  }, [currentJob?.id, currentJob?.status]);

  // Quick 1-click sample loader
  const handleLoadSampleTraffic = () => {
    setActiveTab('studio');
  };

  const handleJumpToTime = (timeSec: number) => {
    // Jump handler passed down to video in studio
  };

  const isProcessing = currentJob?.status === 'extracting' || currentJob?.status === 'analyzing' || currentJob?.status === 'queued';

  return (
    <div className="min-h-screen bg-carbon-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-black">
      
      {/* Header */}
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        systemCaps={systemCaps}
        onOpenSystemModal={() => setIsSystemModalOpen(true)}
        onLoadSample={handleLoadSampleTraffic}
        isProcessing={!!isProcessing}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'studio' && (
          <StudioView
            systemCaps={systemCaps}
            sampleVideos={sampleVideos}
            currentJob={currentJob}
            setCurrentJob={setCurrentJob}
            onJobComplete={(job) => {
              setCurrentJob(job);
              getJobResults(job.id).then((res) => setResultsOverlay(res));
            }}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsView
            currentJob={currentJob}
            resultsOverlay={resultsOverlay}
            onJumpToTime={handleJumpToTime}
            onSwitchToStudio={() => setActiveTab('studio')}
          />
        )}

        {activeTab === 'exports' && (
          <ExportView
            currentJob={currentJob}
            onSwitchToStudio={() => setActiveTab('studio')}
          />
        )}

        {activeTab === 'history' && (
          <HistoryView
            onSelectJob={(job) => {
              setCurrentJob(job);
              getJobResults(job.id).then((res) => setResultsOverlay(res));
            }}
            onSwitchToStudio={() => setActiveTab('studio')}
          />
        )}
      </main>

      {/* System Diagnostics Modal */}
      <SystemModal
        isOpen={isSystemModalOpen}
        onClose={() => setIsSystemModalOpen(false)}
        systemCaps={systemCaps}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-carbon-950/80 py-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>VisionTrack AI Core v1.0 • YOLO11 & Persistent Multi-Object Tracking</span>
          <span className="text-slate-600">Hardware Accelerated CUDA / NVENC Pipelines</span>
        </div>
      </footer>

    </div>
  );
};

export default App;
