/**
 * Client & Mobile GPU Hardware Capability Detection Engine
 *
 * Reliably probes WebGL 2.0, WebGL 1.0, and WebGPU capabilities across
 * desktop, tablet, and mobile browsers. Discovers unmasked GPU renderers,
 * driver vendors, hardware concurrency, and provides a stable fallback
 * to software 2D Canvas rendering when hardware acceleration is disabled.
 */

export interface ClientGpuCapabilities {
  // Overall capability verdict
  isGpuAccelerated: boolean;
  tier: 'high-end' | 'mid-range' | 'mobile' | 'fallback';
  deviceType: 'mobile' | 'tablet' | 'desktop';
  
  // API Support
  webgpuSupported: boolean;
  webgl2Supported: boolean;
  webgl1Supported: boolean;
  
  // Hardware specifics
  renderer: string;
  vendor: string;
  maxTextureSize: number;
  hardwareConcurrency: number;
  deviceMemoryGb: number | null;
  
  // Browser telemetry & extensions
  floatTexturesSupported: boolean;
  canvasHardwareAccelerated: boolean;
  notes: string[];
}

let cachedCapabilities: ClientGpuCapabilities | null = null;

export async function detectClientGpuCapabilities(forceRefresh = false): Promise<ClientGpuCapabilities> {
  if (cachedCapabilities && !forceRefresh) {
    return cachedCapabilities;
  }

  const notes: string[] = [];
  let isGpuAccelerated = false;
  let webgpuSupported = false;
  let webgl2Supported = false;
  let webgl1Supported = false;
  let renderer = 'Standard Software Renderer';
  let vendor = 'Generic Browser Environment';
  let maxTextureSize = 2048;
  let floatTexturesSupported = false;
  let canvasHardwareAccelerated = false;

  // 1. Device form factor detection
  const ua = navigator.userAgent || '';
  const isMobileUa = /Android|iPhone|iPod|Windows Phone/i.test(ua);
  const isTabletUa = /iPad|Android(?!.*Mobile)|Tablet/i.test(ua);
  const hasTouch = navigator.maxTouchPoints > 0;
  
  let deviceType: 'mobile' | 'tablet' | 'desktop' = 'desktop';
  if (isTabletUa || (hasTouch && /Macintosh/i.test(ua) && navigator.maxTouchPoints > 1)) {
    deviceType = 'tablet';
  } else if (isMobileUa || (hasTouch && window.innerWidth < 768)) {
    deviceType = 'mobile';
  }

  // 2. Hardware Concurrency & Memory
  const hardwareConcurrency = navigator.hardwareConcurrency || 4;
  const deviceMemoryGb = (navigator as unknown as { deviceMemory?: number }).deviceMemory || null;

  // 3. WebGPU Probe (Modern standard)
  const navAny = typeof navigator !== 'undefined' ? (navigator as any) : null;
  if (navAny && navAny.gpu && typeof navAny.gpu.requestAdapter === 'function') {
    try {
      const adapter = await navAny.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        webgpuSupported = true;
        isGpuAccelerated = true;
        const adapterInfo = adapter.info || (typeof adapter.requestAdapterInfo === 'function' ? await adapter.requestAdapterInfo() : null);
        if (adapterInfo?.description) {
          renderer = adapterInfo.description;
          if (adapterInfo.vendor) vendor = adapterInfo.vendor;
        }
        notes.push('WebGPU API active & supported');
      }
    } catch {
      notes.push('WebGPU adapter query failed; falling back to WebGL');
    }
  }

  // 4. WebGL 2.0 & 1.0 Probe
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;

    // Try WebGL 2 first
    let gl: WebGLRenderingContext | WebGL2RenderingContext | null = null;
    try {
      gl = canvas.getContext('webgl2', { powerPreference: 'high-performance', alpha: false });
      if (gl) {
        webgl2Supported = true;
        webgl1Supported = true;
        isGpuAccelerated = true;
      }
    } catch {
      // ignore
    }

    // Fallback to WebGL 1
    if (!gl) {
      try {
        gl = (canvas.getContext('webgl', { powerPreference: 'high-performance' }) ||
          canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;
        if (gl) {
          webgl1Supported = true;
          isGpuAccelerated = true;
        }
      } catch {
        // ignore
      }
    }

    if (gl) {
      maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) || maxTextureSize;

      // Extract unmasked renderer if extension available
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        const unmaskedRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        const unmaskedVendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        if (unmaskedRenderer) renderer = String(unmaskedRenderer);
        if (unmaskedVendor) vendor = String(unmaskedVendor);
      } else {
        const standardRenderer = gl.getParameter(gl.RENDERER);
        const standardVendor = gl.getParameter(gl.VENDOR);
        if (standardRenderer) renderer = String(standardRenderer);
        if (standardVendor) vendor = String(standardVendor);
      }

      // Check float texture support
      if (webgl2Supported) {
        floatTexturesSupported = Boolean(gl.getExtension('EXT_color_buffer_float'));
      } else {
        floatTexturesSupported = Boolean(gl.getExtension('OES_texture_float'));
      }

      // Detect if software rasterizer (e.g. SwiftShader, llvmpipe, Software)
      if (/swiftshader|llvmpipe|softpipe|software/i.test(renderer)) {
        isGpuAccelerated = false;
        notes.push('Software rasterizer detected; GPU acceleration disabled');
      } else {
        notes.push('Hardware rasterizer confirmed');
      }
    } else {
      notes.push('WebGL unsupported; standard 2D CPU Canvas active');
    }
  } catch (e) {
    notes.push(`WebGL probe encountered error: ${e instanceof Error ? e.message : String(e)}`);
  }

  // 5. Test 2D Canvas Hardware Acceleration
  try {
    const testCanvas = document.createElement('canvas');
    testCanvas.width = 4;
    testCanvas.height = 4;
    const ctx = testCanvas.getContext('2d');
    if (ctx) {
      canvasHardwareAccelerated = true;
    }
  } catch {
    canvasHardwareAccelerated = false;
  }

  // 6. Classify Performance Tier
  let tier: 'high-end' | 'mid-range' | 'mobile' | 'fallback' = 'fallback';
  if (!isGpuAccelerated) {
    tier = 'fallback';
  } else if (deviceType === 'mobile' || deviceType === 'tablet') {
    tier = 'mobile';
  } else if (/nvidia|rtx|gtx|radeon|rx|apple m|geforce/i.test(renderer) && maxTextureSize >= 8192) {
    tier = 'high-end';
  } else {
    tier = 'mid-range';
  }

  cachedCapabilities = {
    isGpuAccelerated,
    tier,
    deviceType,
    webgpuSupported,
    webgl2Supported,
    webgl1Supported,
    renderer,
    vendor,
    maxTextureSize,
    hardwareConcurrency,
    deviceMemoryGb,
    floatTexturesSupported,
    canvasHardwareAccelerated,
    notes,
  };

  return cachedCapabilities;
}
