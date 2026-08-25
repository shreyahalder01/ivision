export interface SystemCapabilities {
  version: string;
  ready: boolean;
  ai: {
    torchAvailable: boolean;
    torchVersion: string | null;
    ultralyticsAvailable: boolean;
    ultralyticsVersion: string | null;
    cudaAvailable: boolean;
    cudaVersion: string | null;
    device: string;
    deviceName: string;
    vramTotalGb: number | null;
    gpuAccelerated: boolean;
    aiAvailable: boolean;
    reason: string | null;
  };
  ffmpeg: {
    ffmpeg: {
      available: boolean;
      path: string | null;
      version: string | null;
      source: string;
    };
    ffprobe: {
      available: boolean;
      path: string | null;
      version: string | null;
      source: string;
    };
    hardwareEncoders: string[];
    preferredEncoder: string;
    ready: boolean;
  };
  models: Array<{
    key: string;
    label: string;
    speed: string;
    accuracy: string;
    paramsM: number;
    sizeMb: number;
    minVramGb: number;
  }>;
  cocoClasses: Record<string, string>;
  featuredClasses: string[];
  classGroups: Record<string, string[]>;
  trackingMethods: Array<{
    key: string;
    label: string;
    description: string;
  }>;
  annotationStyles: Array<{
    key: string;
    label: string;
  }>;
}

export interface VideoMetadata {
  duration: number;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  codec: string;
  bitrate: number;
  aspect_ratio: string;
  size_bytes?: number;
}

export interface SampleVideo {
  name: string;
  filename: string;
  videoPath: string;
  sizeBytes: number;
  metadata: VideoMetadata;
}

export interface Job {
  id: string;
  filename: string;
  original_name: string;
  status: 'ready' | 'uploading' | 'queued' | 'extracting' | 'analyzing' | 'complete' | 'failed' | 'cancelled';
  created_at: number;
  updated_at: number;
  started_at?: number;
  finished_at?: number;
  video_path: string;
  poster_path?: string;
  sprite_path?: string;
  videoMetadata?: VideoMetadata;
  selectedClasses?: string[];
  confidence: number;
  iou: number;
  model: string;
  tracking_method: string;
  annotation_style: string;
  frame_stride: number;
  progress: number;
  processed_frames: number;
  total_frames: number;
  stage?: string;
  device?: string;
  processing_fps?: number;
  results?: {
    totalDetections: number;
    uniqueObjects: number;
    classCounts: Record<string, number>;
    groupCounts: Record<string, number>;
    averageConfidence: number;
    classDistribution: Array<{
      className: string;
      count: number;
      share: number;
    }>;
    longestTracks: Array<{
      trackId: number;
      className: string;
      duration: number;
      detectionCount: number;
      avgConfidence: number;
      firstSeen: number;
      lastSeen: number;
    }>;
  };
  error?: {
    code: string;
    title: string;
    message: string;
    cause: string;
    action: string;
    detail?: string;
  };
  liveProgress?: {
    processed: number;
    total: number;
    progress: number;
    elapsed: number;
    rate: number;
    etaSeconds: number | null;
  };
}

export interface ResultsOverlayPayload {
  jobId: string;
  schema: string;
  fps: number;
  frameStride: number;
  sourceWidth: number;
  sourceHeight: number;
  partial: boolean;
  fields: string[]; // ["trackId", "classId", "x", "y", "w", "h", "confidence"]
  frames: Record<string, Array<[number, number, number, number, number, number, number]>>;
  tracks: Array<{
    trackId: number;
    classId: number;
    className: string;
    firstFrame: number;
    lastFrame: number;
    firstSeen: number;
    lastSeen: number;
    duration: number;
    detectionCount: number;
    avgConfidence: number;
    maxSpeed: number;
    path: Array<[number, number, number]>;
  }>;
}

export interface TrackRecord {
  job_id: string;
  track_id: number;
  class_id: number;
  class_name: string;
  first_frame: number;
  last_frame: number;
  first_seen: number;
  last_seen: number;
  duration: number;
  detection_count: number;
  avg_confidence: number;
  max_confidence: number;
  max_speed: number;
  avg_speed: number;
  distance: number;
  gap_count: number;
  path: Array<[number, number, number]>;
}

export interface ExportRecord {
  id: string;
  job_id: string;
  kind: 'video' | 'data';
  fmt: string;
  status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled';
  progress: number;
  path?: string;
  size_bytes?: number;
  options?: Record<string, any>;
  created_at: number;
  finished_at?: number;
  error?: {
    code: string;
    title: string;
    message: string;
    cause: string;
    action: string;
    detail?: string;
  };
}
