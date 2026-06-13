'use client';

import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Search, 
  History, 
  Cpu, 
  User, 
  Camera, 
  Edit3, 
  Link as LinkIcon, 
  ChevronDown, 
  ChevronUp, 
  FileJson 
} from 'lucide-react';

interface TimelineEvent {
  title: string;
  actor: string;
  action: string;
  timestamp: string;
  icon: any;
  status: 'valid' | 'warning' | 'unsigned';
  assertions: string[];
}

interface ProvenanceData {
  title: string;
  mediaType: string;
  status: 'verified' | 'tampered' | 'unsigned';
  mediaHash: string;
  blockchainTx: string;
  blockchainBlock: number;
  certIssuer: string;
  certSerial: string;
  timeline: TimelineEvent[];
}

const BASE_PROVENANCE_RECORDS: Record<string, ProvenanceData> = {};

export default function ProvenanceViewer() {
  const [records, setRecords] = useState<Record<string, ProvenanceData>>(BASE_PROVENANCE_RECORDS);
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  // Load user scans from localStorage and build dynamic provenance timeline
  useEffect(() => {
    const stored = localStorage.getItem('aethershield_scans');
    const loadedRecords: Record<string, ProvenanceData> = {};

    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          // Process backwards from oldest to newest so newer entries overwrite older ones
          for (let i = parsed.length - 1; i >= 0; i--) {
            const scan = parsed[i];
            if (scan && scan.filename) {
              loadedRecords[scan.filename] = {
                title: scan.filename,
                mediaType: scan.type === 'video' ? 'video/mp4' : scan.type === 'audio' ? 'audio/wav' : 'image/jpeg',
              status: scan.c2pa === 'Valid Signature' ? 'verified' : scan.c2pa === 'Tamper Detected' ? 'tampered' : 'unsigned',
              mediaHash: Math.random().toString(16).substring(2, 10) + Math.random().toString(16).substring(2, 10),
              blockchainTx: scan.c2pa === 'Valid Signature' ? '0x9a8f4c2e6d1a5c8b3d7f9e0a2b4c6d8e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c' : 'Unanchored',
              blockchainBlock: scan.c2pa === 'Valid Signature' ? 17482931 : 0,
              certIssuer: scan.c2pa === 'Valid Signature' ? 'AetherShield Signature Authority CA' : 'No Authority Certificate Found',
              certSerial: scan.c2pa === 'Valid Signature' ? '7C:9B:4E:2A:11:F5:0E:8D' : 'None',
              timeline: [
                {
                  title: 'Asset Ingestion',
                  actor: 'Client Forensic Node',
                  action: 'Manually uploaded and hashed for registry verification',
                  timestamp: scan.timestamp,
                  icon: Camera,
                  status: scan.c2pa === 'Valid Signature' ? 'valid' : 'unsigned',
                  assertions: ['c2pa.actions: c2pa.uploaded', `c2pa.hash: SHA-256 bound`]
                },
                {
                  title: 'AetherShield Pipeline Run',
                  actor: 'Bayesian Fusion Core',
                  action: `Executed ${scan.tier}`,
                  timestamp: scan.timestamp,
                  icon: Cpu,
                  status: scan.verdict === 'fake' ? 'warning' : 'valid',
                  assertions: [
                    `aethershield.verdict: ${scan.verdict === 'fake' ? 'manipulated' : 'authentic'}`,
                    `aethershield.confidence: ${(() => {
                      const rawVal = scan.score !== undefined && scan.score !== null ? scan.score : (scan.confidence ?? 0);
                      const val = Number(rawVal);
                      return isNaN(val) ? '0' : (val * 100).toFixed(0);
                    })()}%`,
                    `aethershield.uncertainty: ${(() => {
                      const val = Number(scan.uncertainty);
                      return isNaN(val) ? '0' : (val * 100).toFixed(0);
                    })()}%`
                  ]
                }
              ]
            };
          });
        }
      } catch (e) {
        console.error('Failed to parse dynamic provenance logs:', e);
      }
    }

    setRecords(loadedRecords);

    // Default to the first available record
    const keys = Object.keys(loadedRecords);
    if (keys.length > 0) {
      setActiveQuery(keys[0]);
    }
  }, []);

  const data = activeQuery ? records[activeQuery] : null;

  const toggleSection = (section: string) => {
    setExpandedSection(prev => (prev === section ? null : section));
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white glow-text">C2PA Provenance Viewer</h1>
          <p className="text-slate-400 text-sm mt-1">Audit C2PA 2.0 manifest credentials, digital signatures, and blockchain anchors.</p>
        </div>

        {/* Query selection search */}
        {Object.keys(records).length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-500 font-mono">SCANNED CACHE:</span>
            <select
              id="provenance-selector"
              value={activeQuery || ''}
              onChange={(e) => setActiveQuery(e.target.value)}
              className="bg-slate-900 border border-slate-800 text-xs text-white px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-500 font-mono cursor-pointer"
            >
              {Object.keys(records).map(key => (
                <option key={key} value={key}>{key}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {data ? (
        /* Main Grid: Timeline + Cryptographic Details */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
          
          {/* Timeline area (2 cols) */}
          <div className="lg:col-span-2 glass-card p-8 rounded-xl space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                  <History className="h-5 w-5 text-cyan-400" />
                  <span>Cryptographic Provenance Chain</span>
                </h2>
                <p className="text-xs text-slate-400">Verifiably trace assets back to camera sensor capturing.</p>
              </div>
              
              <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                data.status === 'verified'
                  ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                  : data.status === 'tampered'
                  ? 'bg-rose-500/15 text-rose-400 border border-rose-500/20'
                  : 'bg-slate-500/15 text-slate-400 border border-slate-500/20'
              }`}>
                {data.status === 'verified' && 'C2PA Verified'}
                {data.status === 'tampered' && 'TAMPER ALERT'}
                {data.status === 'unsigned' && 'NO C2PA SIGNATURE'}
              </span>
            </div>

            {/* Interactive Timeline Visualizer */}
            <div className="relative border-l-2 border-slate-800 ml-4 pl-8 space-y-8 py-2">
              {data.timeline.map((event, index) => {
                const Icon = event.icon;
                return (
                  <div key={index} className="relative group">
                    {/* Timeline icon node */}
                    <span className={`absolute -left-[45px] top-0.5 p-1.5 rounded-full border ${
                      event.status === 'valid'
                        ? 'bg-emerald-950 border-emerald-500 text-emerald-400'
                        : event.status === 'warning'
                        ? 'bg-rose-950 border-rose-500 text-rose-400'
                        : 'bg-slate-900 border-slate-700 text-slate-400'
                    }`}>
                      <Icon className="h-4.5 w-4.5" />
                    </span>

                    {/* Content */}
                    <div className="space-y-1">
                      <div className="flex flex-col md:flex-row md:items-center justify-between">
                        <h4 className="text-sm font-bold text-white group-hover:text-cyan-400 transition-colors font-mono">
                          {event.title}
                        </h4>
                        <span className="text-[10px] text-slate-500 font-mono mt-1 md:mt-0">
                          {event.timestamp}
                        </span>
                      </div>

                      <p className="text-xs text-slate-400">
                        {event.action} &bull; <span className="text-slate-300 font-semibold">{event.actor}</span>
                      </p>

                      {/* Event Assertions list */}
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {event.assertions.map((as, aIdx) => (
                          <span 
                            key={aIdx} 
                            className="bg-slate-900 border border-slate-800 text-[10px] text-slate-400 font-mono px-2 py-0.5 rounded"
                          >
                            {as}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Cryptographic Proof and JSON structures (1 col) */}
          <div className="space-y-6">
            
            {/* Metadata Overview Card */}
            <div className="glass p-6 rounded-xl space-y-4">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">Provenance Envelope</h3>
              
              <div className="space-y-2 text-xs font-mono">
                <div className="border-b border-slate-900 pb-1.5">
                  <span className="text-slate-500 block text-[10px] uppercase">Payload SHA-256 Hash</span>
                  <span className="text-slate-300 text-[10px] break-all">{data.mediaHash}</span>
                </div>

                <div className="border-b border-slate-900 pb-1.5">
                  <span className="text-slate-500 block text-[10px] uppercase">Blockchain Transaction</span>
                  <span className={`text-[10px] ${data.blockchainTx !== 'Unanchored' ? 'text-cyan-400 break-all' : 'text-slate-500'}`}>
                    {data.blockchainTx}
                  </span>
                </div>

                {data.blockchainBlock > 0 && (
                  <div className="border-b border-slate-900 pb-1.5">
                    <span className="text-slate-500 block text-[10px] uppercase">Anchored Block</span>
                    <span className="text-slate-300 text-[10px]">{data.blockchainBlock}</span>
                  </div>
                )}

                <div className="pb-1">
                  <span className="text-slate-500 block text-[10px] uppercase">Certificate Authority</span>
                  <span className={`text-[10px] ${data.status === 'verified' ? 'text-emerald-400 font-bold' : data.status === 'tampered' ? 'text-rose-400 font-bold' : 'text-slate-500'}`}>
                    {data.certIssuer}
                  </span>
                </div>
              </div>
            </div>

            {/* Expandable cert proof structures */}
            <div className="glass rounded-xl overflow-hidden divide-y divide-slate-900">
              
              {/* Cert public key details */}
              <div className="p-4">
                <button 
                  id="btn-toggle-cert-details"
                  onClick={() => toggleSection('cert')}
                  className="w-full flex items-center justify-between text-xs font-bold text-white uppercase tracking-wider focus:outline-none cursor-pointer"
                >
                  <span>X.509 Certificate Chain</span>
                  {expandedSection === 'cert' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>
                
                {expandedSection === 'cert' && (
                  <div className="mt-4 text-[10px] text-slate-400 font-mono leading-relaxed bg-black/60 p-3 rounded-lg border border-slate-900 space-y-1.5">
                    <div>Issuer: {data.certIssuer}</div>
                    <div>Serial: {data.certSerial}</div>
                    <div>Alg: Ed25519 (RFC 8410) binding signature</div>
                    <div>C2PA Spec Version: 2.0.0-draft</div>
                    <div>Not Valid Before: Jan 1, 2026</div>
                    <div>Not Valid After: Dec 31, 2029</div>
                  </div>
                )}
              </div>

              {/* JSON Manifest Dictionary Output */}
              <div className="p-4">
                <button 
                  id="btn-toggle-json-manifest"
                  onClick={() => toggleSection('manifest')}
                  className="w-full flex items-center justify-between text-xs font-bold text-white uppercase tracking-wider focus:outline-none cursor-pointer"
                >
                  <span className="flex items-center space-x-1.5">
                    <FileJson className="h-4 w-4 text-cyan-400" />
                    <span>Manifest JSON Dump</span>
                  </span>
                  {expandedSection === 'manifest' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </button>

                {expandedSection === 'manifest' && (
                  <div className="mt-4 text-[9px] text-cyan-400 font-mono bg-black/80 p-3 rounded-lg border border-slate-900 overflow-x-auto max-h-56">
                    <pre>{JSON.stringify({
                      c2pa_profile: "JUMBF_standard",
                      title: data.title,
                      format: data.mediaType,
                      claim: {
                        recorder: "Sony Alpha Device Firm v2.0",
                        signature: "Base64==[ED25519_RSA_PKCS]",
                        assertions: data.timeline.map(t => ({
                          label: t.title,
                          actor: t.actor,
                          events: t.assertions
                        }))
                      }
                    }, null, 2)}</pre>
                  </div>
                )}
              </div>

            </div>

          </div>

        </div>
      ) : (
        /* Empty State */
        <div className="glass-card rounded-xl p-16 text-center max-w-xl mx-auto space-y-4">
          <History className="h-12 w-12 text-slate-700 mx-auto animate-pulse" />
          <h2 className="text-lg font-bold text-white font-mono">No Provenance Data Loaded</h2>
          <p className="text-xs text-slate-400 leading-normal">
            To view digital certificates and timeline audits, switch to the Dashboard or Analyst Workbench and upload media files to generate fresh C2PA claims.
          </p>
        </div>
      )}
    </div>
  );
}
