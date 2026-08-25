import React, { useEffect, useRef } from 'react';
import { ResultsOverlayPayload } from '../types';

interface CanvasOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  results: ResultsOverlayPayload | null;
  selectedTrackId: number | null;
  onSelectTrack: (trackId: number | null) => void;
  annotationStyle: string;
  showTrails: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  minConfidenceFilter: number;
}

// Consistent colors for classes/tracks
const CLASS_COLORS: Record<string, string> = {
  person: '#00f0ff', // cyan
  car: '#3b82f6',    // blue
  truck: '#8b5cf6',  // violet
  bus: '#f59e0b',    // amber
  motorcycle: '#ec4899', // pink
  bicycle: '#10b981', // emerald
  dog: '#f97316',    // orange
  cat: '#e11d48',    // rose
};

function getColorForTrack(trackId: number, className: string): string {
  if (CLASS_COLORS[className.toLowerCase()]) {
    return CLASS_COLORS[className.toLowerCase()];
  }
  // Generative distinct hue
  const hue = (trackId * 137.5) % 360;
  return `hsl(${hue}, 85%, 60%)`;
}

export const CanvasOverlay: React.FC<CanvasOverlayProps> = ({
  videoRef,
  results,
  selectedTrackId,
  onSelectTrack,
  annotationStyle,
  showTrails,
  showLabels,
  showConfidence,
  minConfidenceFilter,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || !results) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      // Sync canvas internal resolution to video display bounding box
      const rect = video.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const currentTime = video.currentTime;
      const fps = results.fps || 30.0;
      const currentFrame = Math.round(currentTime * fps);

      // Find frame key in results
      const frameKey = String(currentFrame);
      const detections = results.frames[frameKey] || [];

      const width = canvas.width;
      const height = canvas.height;

      // 1. Draw motion trails if enabled
      if (showTrails && results.tracks) {
        results.tracks.forEach((track) => {
          if (!track.path || track.path.length < 2) return;
          if (selectedTrackId !== null && track.trackId !== selectedTrackId) return;

          // Only show points up to current frame
          const pastPoints = track.path.filter((pt) => pt[0] <= currentFrame);
          if (pastPoints.length < 2) return;

          const color = getColorForTrack(track.trackId, track.className);
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = selectedTrackId === track.trackId ? 3 : 1.5;
          ctx.globalAlpha = selectedTrackId === track.trackId ? 0.9 : 0.45;
          ctx.setLineDash([4, 4]);

          pastPoints.forEach((pt, i) => {
            const px = pt[1] * width;
            const py = pt[2] * height;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1.0;
        });
      }

      // 2. Draw detections for current frame
      detections.forEach((det) => {
        // [trackId, classId, x, y, w, h, confidence]
        const [trackId, , nx, ny, nw, nh, conf] = det;
        if (conf < minConfidenceFilter) return;

        const trackMeta = results.tracks.find((t) => t.trackId === trackId);
        const className = trackMeta ? trackMeta.className : 'object';

        const bx = nx * width;
        const by = ny * height;
        const bw = nw * width;
        const bh = nh * height;

        const color = getColorForTrack(trackId, className);
        const isSelected = selectedTrackId === trackId;

        ctx.save();

        if (annotationStyle === 'mask') {
          // Alpha shaded box
          ctx.fillStyle = isSelected ? 'rgba(0, 240, 255, 0.35)' : `${color}25`;
          ctx.fillRect(bx, by, bw, bh);
          ctx.strokeStyle = color;
          ctx.lineWidth = isSelected ? 3 : 1.5;
          ctx.strokeRect(bx, by, bw, bh);
        } else if (annotationStyle === 'corner_brackets') {
          // Corner brackets
          const cl = Math.min(16, bw / 3, bh / 3);
          ctx.strokeStyle = isSelected ? '#00f0ff' : color;
          ctx.lineWidth = isSelected ? 3 : 2;

          // Top Left
          ctx.beginPath();
          ctx.moveTo(bx, by + cl);
          ctx.lineTo(bx, by);
          ctx.lineTo(bx + cl, by);
          ctx.stroke();

          // Top Right
          ctx.beginPath();
          ctx.moveTo(bx + bw - cl, by);
          ctx.lineTo(bx + bw, by);
          ctx.lineTo(bx + bw, by + cl);
          ctx.stroke();

          // Bottom Left
          ctx.beginPath();
          ctx.moveTo(bx, by + bh - cl);
          ctx.lineTo(bx, by + bh);
          ctx.lineTo(bx + cl, by + bh);
          ctx.stroke();

          // Bottom Right
          ctx.beginPath();
          ctx.moveTo(bx + bw - cl, by + bh);
          ctx.lineTo(bx + bw, by + bh);
          ctx.lineTo(bx + bw, by + bh - cl);
          ctx.stroke();
        } else if (annotationStyle === 'cyber_hud') {
          // Cyber HUD styling
          ctx.strokeStyle = color;
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          ctx.strokeRect(bx, by, bw, bh);

          // Center target reticle
          const cx = bx + bw / 2;
          const cy = by + bh / 2;
          ctx.beginPath();
          ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        } else if (annotationStyle === 'minimal_dot') {
          // Minimal Center Dot
          const cx = bx + bw / 2;
          const cy = by + bh / 2;
          ctx.beginPath();
          ctx.arc(cx, cy, isSelected ? 6 : 4, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        } else {
          // Standard box_label
          ctx.strokeStyle = isSelected ? '#00f0ff' : color;
          ctx.lineWidth = isSelected ? 3 : 2;
          if (isSelected) {
            ctx.shadowColor = '#00f0ff';
            ctx.shadowBlur = 10;
          }
          ctx.strokeRect(bx, by, bw, bh);
        }

        // Draw Labels / Badges
        if (showLabels && annotationStyle !== 'minimal_dot') {
          const confText = showConfidence ? ` ${(conf * 100).toFixed(0)}%` : '';
          const labelText = `#${trackId} ${className}${confText}`;

          ctx.font = '600 11px JetBrains Mono, monospace';
          const textWidth = ctx.measureText(labelText).width;
          const padX = 6;
          const padY = 4;
          const badgeHeight = 18;
          const badgeWidth = textWidth + padX * 2;

          let badgeX = bx;
          let badgeY = by - badgeHeight - 2;
          if (badgeY < 0) badgeY = by + 2;

          // Badge background
          ctx.fillStyle = 'rgba(11, 15, 25, 0.9)';
          ctx.fillRect(badgeX, badgeY, badgeWidth, badgeHeight);

          // Badge accent border
          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.strokeRect(badgeX, badgeY, badgeWidth, badgeHeight);

          // Text
          ctx.fillStyle = isSelected ? '#00f0ff' : '#ffffff';
          ctx.fillText(labelText, badgeX + padX, badgeY + 13);
        }

        ctx.restore();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [videoRef, results, selectedTrackId, annotationStyle, showTrails, showLabels, showConfidence, minConfidenceFilter]);

  // Click on canvas to select track
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !results) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const currentTime = video.currentTime;
    const fps = results.fps || 30.0;
    const currentFrame = Math.round(currentTime * fps);
    const frameKey = String(currentFrame);
    const detections = results.frames[frameKey] || [];

    const width = canvas.width;
    const height = canvas.height;

    let clickedTrackId: number | null = null;

    // Check hit test against current frame detections (topmost first)
    for (let i = detections.length - 1; i >= 0; i--) {
      const [trackId, , nx, ny, nw, nh] = detections[i];
      const bx = nx * width;
      const by = ny * height;
      const bw = nw * width;
      const bh = nh * height;

      if (clickX >= bx && clickX <= bx + bw && clickY >= by && clickY <= by + bh) {
        clickedTrackId = trackId;
        break;
      }
    }

    onSelectTrack(clickedTrackId === selectedTrackId ? null : clickedTrackId);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleCanvasClick}
      className="absolute inset-0 w-full h-full pointer-events-auto cursor-crosshair"
    />
  );
};
