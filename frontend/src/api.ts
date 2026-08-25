import {
  SystemCapabilities,
  SampleVideo,
  Job,
  ResultsOverlayPayload,
  TrackRecord,
  ExportRecord,
} from './types';

const API_BASE = '/api';

export async function fetchCapabilities(): Promise<SystemCapabilities> {
  const res = await fetch(`${API_BASE}/system/capabilities`);
  if (!res.ok) throw new Error('Failed to fetch system capabilities');
  return res.json();
}

export async function fetchSampleVideos(): Promise<SampleVideo[]> {
  const res = await fetch(`${API_BASE}/videos/samples`);
  if (!res.ok) throw new Error('Failed to fetch sample videos');
  return res.json();
}

export async function uploadVideo(
  file: File,
  onProgress?: (percent: number) => void
): Promise<{
  tempId: string;
  filename: string;
  videoPath: string;
  sizeBytes: number;
  metadata: any;
  posterPath: string | null;
  spritePath: string | null;
  spriteMeta: any;
}> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    xhr.open('POST', `${API_BASE}/videos/upload`);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable) {
          onProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error('Invalid response from server'));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail?.message || err.detail || 'Upload failed'));
        } catch {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(formData);
  });
}

export async function createJob(payload: {
  video_path: string;
  original_name?: string;
  classes: string[];
  confidence: number;
  iou: number;
  model: string;
  tracking_method: string;
  annotation_style: string;
  frame_stride: number;
}): Promise<Job> {
  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail?.message || err.detail || 'Failed to create job');
  }
  return res.json();
}

export async function listJobs(limit = 50, offset = 0, status?: string): Promise<{
  items: Job[];
  total: number;
}> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) params.append('status', status);
  const res = await fetch(`${API_BASE}/jobs?${params.toString()}`);
  if (!res.ok) throw new Error('Failed to list jobs');
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('Failed to fetch job');
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to cancel job');
}

export async function resumeJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/resume`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to resume job');
}

export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete job');
}

export async function getJobResults(jobId: string): Promise<ResultsOverlayPayload> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/results`);
  if (!res.ok) throw new Error('Failed to fetch job results');
  return res.json();
}

export async function getJobTracks(jobId: string): Promise<TrackRecord[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/tracks`);
  if (!res.ok) throw new Error('Failed to fetch tracks');
  return res.json();
}

export async function createExport(
  jobId: string,
  format: string,
  options: Record<string, any> = {}
): Promise<ExportRecord> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/exports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format, options }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail?.message || err.detail || 'Failed to start export');
  }
  return res.json();
}

export async function listJobExports(jobId: string): Promise<ExportRecord[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/exports`);
  if (!res.ok) throw new Error('Failed to list exports');
  return res.json();
}

export function getMediaUrl(pathOrFilename: string): string {
  return `${API_BASE}/media/${encodeURIComponent(pathOrFilename)}`;
}

export function getExportDownloadUrl(exportId: string): string {
  return `${API_BASE}/exports/${exportId}/download`;
}

export function connectJobWebSocket(
  jobId: string,
  onEvent: (event: any) => void
): () => void {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/jobs/${jobId}`;
  let ws: WebSocket | null = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch (e) {
      console.error('WebSocket parse error', e);
    }
  };

  ws.onerror = (e) => {
    console.debug('WebSocket error', e);
  };

  return () => {
    if (ws) {
      ws.close();
      ws = null;
    }
  };
}
