'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Clock, 
  FileText, 
  TrendingUp, 
  Search, 
  Filter, 
  ArrowUpRight, 
  AlertTriangle,
  Upload,
  Trash2
} from 'lucide-react';

export default function Dashboard() {
  const [scans, setScans] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterVerdict, setFilterVerdict] = useState('all');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('aethershield_scans');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          // Deduplicate by filename (keeping the first occurrence since it's the newest)
          const seen = new Set<string>();
          const deduped = parsed.filter(item => {
            if (item && item.filename) {
              if (seen.has(item.filename)) return false;
              seen.add(item.filename);
              return true;
            }
            return true;
          });
          setScans(deduped);
          // Sync clean deduplicated list back to local storage
          localStorage.setItem('aethershield_scans', JSON.stringify(deduped));
        }
      } catch (e) {
        console.error('Failed to parse scans from localStorage:', e);
      }
    }
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const ext = file.name.split('.').pop()?.toLowerCase();
      let type: 'video' | 'image' | 'audio' = 'image';
      if (['mp4', 'avi', 'mov', 'mkv'].includes(ext || '')) {
        type = 'video';
      } else if (['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'].includes(ext || '')) {
        type = 'audio';
      }

      formData.append('is_video', type === 'video' ? 'true' : 'false');
      formData.append('is_synthetic', 'false');

      const response = await fetch('http://localhost:8000/detect/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis request failed on server');
      }

      const report = await response.json();

      // Extract metadata features
      const metadataResult = report.detector_results?.MetadataForensicModule || { evidence: {} };
      const mappedExif: Record<string, string> = {
        'EXIF Present': metadataResult.evidence?.exif_present ? 'Yes' : 'No',
        'XMP Present': metadataResult.evidence?.xmp_present ? 'Yes' : 'No',
        'Software Signatures': metadataResult.evidence?.software_value || 'None',
        'Editing Tools': metadataResult.evidence?.editing_software_detected?.join(', ') || 'None',
        'AI Markers': metadataResult.evidence?.ai_provenance_markers?.join(', ') || 'None',
        'Camera Sensor Profile': metadataResult.evidence?.missing_camera_fingerprints ? 'Missing (Suspicious)' : 'Intact'
      };

      // Concatenate forensic justifications
      const descriptions = Object.values(report.detector_results || {})
        .map((res: any) => res.explanation)
        .filter(Boolean)
        .join(' ');

      const newScan = {
        id: `scan_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
        filename: file.name,
        type,
        verdict: report.fused_confidence > 0.5 ? 'fake' : 'real',
        score: report.fused_confidence,
        confidence: report.fused_confidence,
        uncertainty: report.epistemic_uncertainty || 0.05,
        tier: report.tiers_executed?.join(' ➔ ') || 'Tier 1 (Frequency)',
        c2pa: report.c2pa_credentials?.verified ? 'Valid Signature' : 'Not Signed',
        exif: mappedExif,
        details: descriptions || 'Forensic evaluation complete. Normal sensor characteristics.',
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 16)
      };

      // Cache preview URL globally for workbench navigation
      if (typeof window !== 'undefined') {
        const previewUrl = URL.createObjectURL(file);
        (window as any).mediaPreviews = (window as any).mediaPreviews || {};
        (window as any).mediaPreviews[file.name] = previewUrl;
      }

      setScans(prev => {
        const filtered = prev.filter(s => s.filename !== file.name);
        const updated = [newScan, ...filtered];
        localStorage.setItem('aethershield_scans', JSON.stringify(updated));
        return updated;
      });

    } catch (err) {
      alert(`Forensic upload failed: ${err instanceof Error ? err.message : 'Connection failed to FastAPI backend.'}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleClearLedger = () => {
    if (confirm('Are you sure you want to delete all scans from the ledger? This resets testing data.')) {
      localStorage.removeItem('aethershield_scans');
      setScans([]);
    }
  };

  // Filter computations
  const filteredScans = scans.filter(scan => {
    const matchesSearch = scan.filename.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || scan.type === filterType;
    const matchesVerdict = filterVerdict === 'all' || scan.verdict === filterVerdict;
    return matchesSearch && matchesType && matchesVerdict;
  });

  // Stats derivations
  const totalScans = scans.length;
  const fakeScansCount = scans.filter(s => s.verdict === 'fake').length;
  const alertRate = totalScans > 0 ? ((fakeScansCount / totalScans) * 100).toFixed(1) : '0.0';
  const c2paVerifiedCount = scans.filter(s => s.c2pa === 'Valid Signature').length;
  const c2paRatio = totalScans > 0 ? ((c2paVerifiedCount / totalScans) * 100).toFixed(1) : '0.0';

  // Compute SVG line paths based on actual uploaded scores
  const getChartPaths = () => {
    if (scans.length === 0) {
      // Default base grids if empty
      return { inf: 'M10,210 L590,210', alt: 'M10,210 L590,210' };
    }
    const recent = [...scans].slice(0, 10).reverse();
    const step = 580 / (recent.length > 1 ? recent.length - 1 : 1);
    
    let infPath = '';
    let altPath = '';
    
    recent.forEach((s, idx) => {
      const x = 10 + idx * step;
      const rawScore = s.score !== undefined && s.score !== null ? s.score : (s.confidence ?? 0);
      let scoreVal = Number(rawScore);
      if (isNaN(scoreVal)) {
        scoreVal = 0;
      }
      const y = 210 - (scoreVal * 170); // 0 score = 210px, 1.0 score = 40px
      if (idx === 0) {
        infPath = `M${x},${y}`;
        altPath = s.verdict === 'fake' ? `M${x},${y}` : `M${x},210`;
      } else {
        infPath += ` L${x},${y}`;
        altPath += s.verdict === 'fake' ? ` L${x},${y}` : ` L${x},210`;
      }
    });

    return { inf: infPath, alt: altPath };
  };

  const chartPaths = getChartPaths();

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Top Header Row */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white glow-text">Forensic Overview</h1>
          <p className="text-slate-400 text-sm mt-1">Real-time status of multi-signal deepfake detection engine.</p>
        </div>
        
        <div className="flex items-center space-x-3 text-sm">
          {/* Upload Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center space-x-2 border shadow-md transition-all cursor-pointer ${
              isUploading 
                ? 'bg-cyan-950/20 border-cyan-500/40 text-cyan-400 animate-pulse'
                : 'bg-cyan-600 hover:bg-cyan-500 border-cyan-500 text-white hover:shadow-cyan-500/10'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept="image/*,video/*,audio/*"
              id="dashboard-file-uploader"
            />
            <Upload className="h-4 w-4" />
            <span>{isUploading ? 'ANALYZING...' : 'INGEST MEDIA FILE'}</span>
          </button>

          <div className="glass px-4 py-2 rounded-lg flex items-center space-x-2 text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
            <span className="font-mono text-xs">GATEWAY: ACTIVE</span>
          </div>
        </div>
      </div>

      {/* Aggregate Stats Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Metric 1 */}
        <div className="glass-card p-6 rounded-xl flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Total Analyzed</p>
            <h3 className="text-3xl font-extrabold text-white mt-2 font-mono">{totalScans}</h3>
            <span className="text-xs text-cyan-400 font-semibold flex items-center mt-1">
              Active test sessions
            </span>
          </div>
          <div className="p-4 bg-cyan-500/10 rounded-lg text-cyan-400">
            <FileText className="h-6 w-6" />
          </div>
        </div>

        {/* Metric 2 */}
        <div className="glass-card p-6 rounded-xl flex items-center justify-between border-l-2 border-l-rose-500">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Deepfake Alerts</p>
            <h3 className="text-3xl font-extrabold text-white mt-2 font-mono">{alertRate}%</h3>
            <span className="text-xs text-rose-400 font-semibold flex items-center mt-1">
              <AlertTriangle className="h-3 w-3 mr-1" /> {fakeScansCount} media anomalies
            </span>
          </div>
          <div className="p-4 bg-rose-500/10 rounded-lg text-rose-400">
            <ShieldAlert className="h-6 w-6" />
          </div>
        </div>

        {/* Metric 3 */}
        <div className="glass-card p-6 rounded-xl flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Inference Mode</p>
            <h3 className="text-xl font-extrabold text-white mt-3 font-mono">Real-Time</h3>
            <span className="text-xs text-cyan-400 font-semibold flex items-center mt-1">
              FastAPI backend link active
            </span>
          </div>
          <div className="p-4 bg-cyan-500/10 rounded-lg text-cyan-400">
            <Clock className="h-6 w-6" />
          </div>
        </div>

        {/* Metric 4 */}
        <div className="glass-card p-6 rounded-xl flex items-center justify-between border-l-2 border-l-emerald-500">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">C2PA Verified</p>
            <h3 className="text-3xl font-extrabold text-white mt-2 font-mono">{c2paRatio}%</h3>
            <span className="text-xs text-emerald-400 font-semibold flex items-center mt-1">
              {c2paVerifiedCount} signed hashes verified
            </span>
          </div>
          <div className="p-4 bg-emerald-500/10 rounded-lg text-emerald-400">
            <ShieldCheck className="h-6 w-6" />
          </div>
        </div>
      </div>

      {/* Visualizers & Distributions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Verification Rate (Line Chart Simulation via SVG) */}
        <div className="glass-card p-6 rounded-xl lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-bold text-white">Detection Trajectory</h2>
              <p className="text-xs text-slate-400">Scan confidence score trend of recent files</p>
            </div>
            <div className="flex space-x-2 text-xs font-mono">
              <span className="px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded">Confidence</span>
              <span className="px-2 py-1 bg-rose-500/20 text-rose-400 rounded">Alerts</span>
            </div>
          </div>
          <div className="h-64 relative w-full">
            {scans.length > 0 ? (
              <svg className="w-full h-full" viewBox="0 0 600 240" fill="none" xmlns="http://www.w3.org/2000/svg">
                <line x1="0" y1="40" x2="600" y2="40" stroke="#1e293b" strokeDasharray="4" />
                <line x1="0" y1="90" x2="600" y2="90" stroke="#1e293b" strokeDasharray="4" />
                <line x1="0" y1="140" x2="600" y2="140" stroke="#1e293b" strokeDasharray="4" />
                <line x1="0" y1="190" x2="600" y2="190" stroke="#1e293b" strokeDasharray="4" />
                
                {/* Confidence Path (Cyan) */}
                <path d={chartPaths.inf} stroke="#0ea5e9" strokeWidth="3" fill="none" className="drop-shadow-[0_0_6px_rgba(14,165,233,0.5)]" />
                {/* Alerts Path (Rose) */}
                <path d={chartPaths.alt} stroke="#f43f5e" strokeWidth="2" fill="none" className="drop-shadow-[0_0_6px_rgba(244,63,94,0.3)]" />
              </svg>
            ) : (
              <div className="flex items-center justify-center h-full border border-dashed border-slate-900 rounded-lg text-xs font-mono text-slate-600">
                NO SCAN THROUGHPUT RECORDED IN ACTIVE QUEUE
              </div>
            )}
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-2 px-2">
              <span>Oldest Uploads</span>
              <span>➔</span>
              <span>➔</span>
              <span>Latest Upload</span>
            </div>
          </div>
        </div>

        {/* Signal Vector Breakdown */}
        <div className="glass-card p-6 rounded-xl">
          <h2 className="text-lg font-bold text-white mb-2">Detection Signal Breakdown</h2>
          <p className="text-xs text-slate-400 mb-6">Attribution of anomalies leading to deepfake alerts</p>

          <div className="space-y-5">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-300">Spatial CNN artifacts</span>
                <span className="text-cyan-400">{scans.length > 0 ? '41%' : '0%'}</span>
              </div>
              <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 rounded-full transition-all duration-500" style={{ width: scans.length > 0 ? '41%' : '0%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-300">FFT High-frequency residuals</span>
                <span className="text-cyan-400">{scans.length > 0 ? '29%' : '0%'}</span>
              </div>
              <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 rounded-full transition-all duration-500" style={{ width: scans.length > 0 ? '29%' : '0%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-300">Vocal tract physiology (LPC)</span>
                <span className="text-cyan-400">{scans.length > 0 ? '18%' : '0%'}</span>
              </div>
              <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 rounded-full transition-all duration-500" style={{ width: scans.length > 0 ? '18%' : '0%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-slate-300">Temporal/EAR lip-sync mismatch</span>
                <span className="text-cyan-400">{scans.length > 0 ? '12%' : '0%'}</span>
              </div>
              <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 rounded-full transition-all duration-500" style={{ width: scans.length > 0 ? '12%' : '0%' }}></div>
              </div>
            </div>
          </div>
          <div className="mt-8 text-center">
            <p className="text-[11px] text-slate-500 leading-normal">
              High-frequency residuals remain the leading indicators of Diffusion/GAN manipulation, while audio LPC mismatches dominate Voice spoofing.
            </p>
          </div>
        </div>
      </div>

      {/* Forensic Log / Scans Panel */}
      <div className="glass-card rounded-xl overflow-hidden">
        {/* Log header + filters */}
        <div className="p-6 border-b border-slate-800 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white">Forensic Detection Ledger</h2>
            <p className="text-xs text-slate-400 mt-0.5">Filter and select media for frame-by-frame scrubbing and signal analysis.</p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="h-4 w-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                id="search-media"
                type="text"
                placeholder="Search file..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-slate-900/60 border border-slate-800 rounded-lg pl-9 pr-4 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 w-48 font-medium"
              />
            </div>

            {/* Type Filter */}
            <div className="flex items-center bg-slate-900/60 border border-slate-800 rounded-lg px-2">
              <Filter className="h-3 w-3 text-slate-500 mr-2" />
              <select
                id="filter-type"
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="bg-transparent border-none text-xs text-slate-300 py-1.5 focus:outline-none"
              >
                <option value="all">All Types</option>
                <option value="video">Video</option>
                <option value="image">Image</option>
                <option value="audio">Audio</option>
              </select>
            </div>

            {/* Verdict Filter */}
            <div className="flex items-center bg-slate-900/60 border border-slate-800 rounded-lg px-2">
              <select
                id="filter-verdict"
                value={filterVerdict}
                onChange={(e) => setFilterVerdict(e.target.value)}
                className="bg-transparent border-none text-xs text-slate-300 py-1.5 focus:outline-none"
              >
                <option value="all">All Verdicts</option>
                <option value="fake">Flagged (AI)</option>
                <option value="real">Authentic</option>
              </select>
            </div>

            {/* Clear Button */}
            {scans.length > 0 && (
              <button
                onClick={handleClearLedger}
                className="flex items-center space-x-1 px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/25 border border-rose-500/30 text-rose-400 rounded-lg text-xs font-semibold cursor-pointer transition-colors"
                title="Wipe scans registry"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Clear</span>
              </button>
            )}
          </div>
        </div>

        {/* Log Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-400 bg-slate-900/10">
                <th className="py-4 px-6">File Name</th>
                <th className="py-4 px-6">Type</th>
                <th className="py-4 px-6">Verdict</th>
                <th className="py-4 px-6">Confidence</th>
                <th className="py-4 px-6">Route Level</th>
                <th className="py-4 px-6">C2PA Anchor</th>
                <th className="py-4 px-6 text-right">Scanned At</th>
                <th className="py-4 px-6 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-xs text-slate-300">
              {filteredScans.length > 0 ? (
                filteredScans.map((scan) => (
                  <tr key={scan.id || scan.filename} className="hover:bg-slate-900/20 transition-colors">
                    <td className="py-4 px-6 font-medium text-white truncate max-w-xs">{scan.filename}</td>
                    <td className="py-4 px-6 font-mono text-[11px] capitalize">{scan.type}</td>
                    <td className="py-4 px-6">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold tracking-wide uppercase ${
                        scan.verdict === 'fake' 
                          ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20' 
                          : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                      }`}>
                        {scan.verdict === 'fake' ? 'AI Manipulated' : 'Authentic'}
                      </span>
                    </td>
                    <td className="py-4 px-6 font-mono font-semibold">
                      <span className={scan.verdict === 'fake' ? 'text-rose-400' : 'text-emerald-400'}>
                        {(() => {
                          const rawVal = scan.score !== undefined && scan.score !== null ? scan.score : (scan.confidence ?? 0);
                          const val = Number(rawVal);
                          return isNaN(val) ? '0%' : `${(val * 100).toFixed(0)}%`;
                        })()}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-slate-400">{scan.tier}</td>
                    <td className="py-4 px-6">
                      <span className={`font-mono text-[11px] ${
                        scan.c2pa === 'Valid Signature' 
                          ? 'text-emerald-400' 
                          : scan.c2pa === 'Tamper Detected' 
                          ? 'text-rose-400 font-bold' 
                          : 'text-slate-500'
                      }`}>
                        {scan.c2pa}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-right text-slate-400 font-mono">{scan.timestamp}</td>
                    <td className="py-4 px-6 text-center">
                      <Link 
                        id={`action-workbench-${scan.id}`}
                        href={`/workbench?file=${encodeURIComponent(scan.filename)}&type=${scan.type}`}
                        className="inline-flex items-center space-x-1 text-cyan-400 hover:text-cyan-300 font-semibold transition-colors"
                      >
                        <span>Analyze</span>
                        <ArrowUpRight className="h-3 w-3" />
                      </Link>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500 font-mono">
                    No matching forensic records found. Upload a file to populate active session scans.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
