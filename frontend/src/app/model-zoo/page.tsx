'use client';

import React, { useState } from 'react';
import { 
  Database, 
  Activity, 
  GitBranch, 
  TrendingUp, 
  Play, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldCheck, 
  RefreshCw 
} from 'lucide-react';

interface ModelHistory {
  version: string;
  accuracy: number;
  auc: number;
  fpr: number;
  status: 'active' | 'retired';
  timestamp: string;
  config?: string;
}

interface DetectorModel {
  name: string;
  displayName: string;
  activeVersion: string;
  type: string;
  accuracy: number;
  auc: number;
  fpr: number;
  alert: boolean;
  history: ModelHistory[];
}

const INITIAL_MODELS: DetectorModel[] = [
  {
    name: 'FrequencyDomainAnalyzer',
    displayName: 'Frequency Domain FFT Analyzer',
    activeVersion: 'v2.1.0',
    type: 'Signal Forensics',
    accuracy: 0.92,
    auc: 0.93,
    fpr: 0.03,
    alert: true,
    history: [
      { version: 'v2.1.0', accuracy: 0.92, auc: 0.93, fpr: 0.03, status: 'active', timestamp: '2026-05-12' },
      { version: 'v2.0.0', accuracy: 0.94, auc: 0.95, fpr: 0.02, status: 'retired', timestamp: '2026-04-12' }
    ]
  },
  {
    name: 'SpatialCNNDetector',
    displayName: 'Spatial EfficientNet CNN',
    activeVersion: 'v4.3.0',
    type: 'Neural Classifier',
    accuracy: 0.93,
    auc: 0.94,
    fpr: 0.022,
    alert: false,
    history: [
      { version: 'v4.3.0', accuracy: 0.93, auc: 0.94, fpr: 0.022, status: 'active', timestamp: '2026-05-15' },
      { version: 'v4.2.0', accuracy: 0.95, auc: 0.96, fpr: 0.015, status: 'retired', timestamp: '2026-04-15' }
    ]
  },
  {
    name: 'TemporalCoherenceAnalyzer',
    displayName: 'Temporal Coherence Tracker',
    activeVersion: 'v1.8.0',
    type: 'Sequence Analysis',
    accuracy: 0.91,
    auc: 0.925,
    fpr: 0.04,
    alert: false,
    history: [
      { version: 'v1.8.0', accuracy: 0.91, auc: 0.925, fpr: 0.04, status: 'active', timestamp: '2026-05-20' }
    ]
  },
  {
    name: 'AudioDeepfakeDetector',
    displayName: 'Audio ResNet-18 Spoof Detector',
    activeVersion: 'v3.0.0',
    type: 'Spectrogram Classifier',
    accuracy: 0.94,
    auc: 0.96,
    fpr: 0.01,
    alert: false,
    history: [
      { version: 'v3.0.0', accuracy: 0.94, auc: 0.96, fpr: 0.01, status: 'active', timestamp: '2026-05-25' }
    ]
  },
  {
    name: 'MetadataForensicModule',
    displayName: 'Metadata Forensics Engine',
    activeVersion: 'v1.0.0',
    type: 'Header Forensics',
    accuracy: 0.98,
    auc: 0.99,
    fpr: 0.005,
    alert: false,
    history: [
      { version: 'v1.0.0', accuracy: 0.98, auc: 0.99, fpr: 0.005, status: 'active', timestamp: '2026-05-01' }
    ]
  }
];

export default function ModelZoo() {
  const [models, setModels] = useState<DetectorModel[]>(INITIAL_MODELS);
  const [trainingModel, setTrainingModel] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);

  const handleRetrain = (modelName: string) => {
    if (trainingModel) return;
    setTrainingModel(modelName);
    setTrainingProgress(0);
    setTrainingLog(['Initializing PGD-10 Adversarial Training loop...', 'Loading dataset build: 2026.06.12-adv', 'Applying Mixed Precision (FP16)...']);

    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        const next = prev + 10;
        if (next === 30) {
          setTrainingLog(log => [...log, 'Epoch 1/5 - Loss: 0.245, Accuracy: 0.912', 'Epoch 2/5 - Loss: 0.198, Accuracy: 0.934']);
        }
        if (next === 60) {
          setTrainingLog(log => [...log, 'Epoch 3/5 - Loss: 0.155, Accuracy: 0.951', 'Epoch 4/5 - Loss: 0.118, Accuracy: 0.968']);
        }
        if (next === 90) {
          setTrainingLog(log => [...log, 'Epoch 5/5 - Loss: 0.092, Accuracy: 0.974', 'Running Canary verification suite...']);
        }
        if (next >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            // Update the model metrics in UI state
            setModels(prevModels => 
              prevModels.map(m => {
                if (m.name === modelName) {
                  const major = parseInt(m.activeVersion.split('.')[0].replace('v', ''));
                  const minor = parseInt(m.activeVersion.split('.')[1]);
                  const newVer = `v${major}.${minor + 1}.0`;
                  const newEntry: ModelHistory = {
                    version: newVer,
                    accuracy: 0.965,
                    auc: 0.978,
                    fpr: 0.012,
                    status: 'active',
                    timestamp: '2026-06-12'
                  };
                  return {
                    ...m,
                    activeVersion: newVer,
                    accuracy: 0.965,
                    auc: 0.978,
                    fpr: 0.012,
                    alert: false,
                    history: [newEntry, ...m.history.map(h => ({ ...h, status: 'retired' as const }))]
                  };
                }
                return m;
              })
            );
            setTrainingModel(null);
          }, 800);
          return 100;
        }
        return next;
      });
    }, 400);
  };

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white glow-text">Model Registry & Zoo</h1>
          <p className="text-slate-400 text-sm mt-1">Manage, evaluate, and trigger continuous retraining loops for deepfake detectors.</p>
        </div>
        <div className="glass px-4 py-2 rounded-lg flex items-center space-x-2 text-slate-300 font-mono text-xs">
          <Database className="h-4 w-4 text-cyan-400" />
          <span>REGISTRY: ANCHORED</span>
        </div>
      </div>

      {/* Main Grid: Left is registry cards, right is training monitor console */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left list: detector cards (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          {models.map(model => (
            <div 
              key={model.name}
              className={`glass-card p-6 rounded-xl relative overflow-hidden transition-all duration-300 ${
                model.alert ? 'border border-rose-500/30 bg-rose-950/5' : ''
              }`}
            >
              {/* Top Row */}
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest font-mono">
                    {model.type}
                  </span>
                  <h3 className="text-lg font-bold text-white mt-1">{model.displayName}</h3>
                  <span className="text-xs font-mono text-slate-500">{model.name}</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className="bg-slate-900 border border-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded-full font-mono">
                    {model.activeVersion}
                  </span>
                  {model.alert && (
                    <span className="bg-rose-500/15 border border-rose-500/20 text-rose-400 text-[10px] font-bold px-2 py-0.5 rounded flex items-center space-x-1 animate-pulse">
                      <AlertTriangle className="h-3 w-3" />
                      <span>DRIFT DETECTED</span>
                    </span>
                  )}
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-4 mt-6 p-4 bg-slate-950/20 rounded-lg border border-slate-900/40">
                <div className="text-center md:text-left">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">ROC-AUC</span>
                  <p className="text-xl font-extrabold text-white mt-1 font-mono">{(model.auc * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center md:text-left">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Accuracy</span>
                  <p className="text-xl font-extrabold text-white mt-1 font-mono">{(model.accuracy * 100).toFixed(1)}%</p>
                </div>
                <div className="text-center md:text-left">
                  <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">False Positive Rate</span>
                  <p className={`text-xl font-extrabold mt-1 font-mono ${model.fpr > 0.025 ? 'text-amber-400' : 'text-slate-300'}`}>
                    {(model.fpr * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              {/* Action and Registry details */}
              <div className="flex items-center justify-between mt-6 text-xs border-t border-slate-900 pt-4">
                <div className="flex items-center space-x-4 text-slate-500 font-mono text-[10px]">
                  <span>HISTORY: {model.history.length} builds</span>
                  <span>LAST DRIFT EVAL: {model.history[0]?.timestamp || 'N/A'}</span>
                </div>

                <button
                  id={`btn-retrain-${model.name}`}
                  onClick={() => handleRetrain(model.name)}
                  disabled={!!trainingModel}
                  className={`px-3 py-1.5 rounded text-xs font-bold transition-all flex items-center space-x-1 border ${
                    model.alert 
                      ? 'bg-rose-500/10 border-rose-500/40 hover:bg-rose-500/20 text-rose-300' 
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700 text-slate-300'
                  } disabled:opacity-50`}
                >
                  <RefreshCw className={`h-3 w-3 ${trainingModel === model.name ? 'animate-spin' : ''}`} />
                  <span>{trainingModel === model.name ? 'TRAINING...' : 'RETRAIN MODEL'}</span>
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Right column: Training monitoring log console (1 col) */}
        <div className="glass-card p-6 rounded-xl flex flex-col h-full min-h-[400px]">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
            <Activity className="h-4 w-4 text-cyan-400" />
            <span>Retraining Telemetry</span>
          </h3>
          <p className="text-xs text-slate-400 mt-1">PGD adversarial augmentation and mixed-precision compilation scheduler.</p>

          <div className="flex-1 bg-black/80 rounded-lg border border-slate-900/80 p-4 mt-4 font-mono text-xs text-slate-300 overflow-y-auto space-y-3 flex flex-col justify-between">
            {trainingModel ? (
              <div className="space-y-4">
                {/* Active progress */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>COMPILING MODEL: {trainingModel}</span>
                    <span className="text-cyan-400">{trainingProgress}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-950 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-cyan-500 rounded-full transition-all duration-300"
                      style={{ width: `${trainingProgress}%` }}
                    ></div>
                  </div>
                </div>

                {/* Log list */}
                <div className="space-y-2 text-[10px] text-cyan-400">
                  {trainingLog.map((log, idx) => (
                    <div key={idx} className="flex items-start space-x-1">
                      <span className="text-slate-600">&gt;</span>
                      <span className="leading-relaxed">{log}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full py-12 text-slate-500 text-center space-y-2">
                <ShieldCheck className="h-8 w-8 text-slate-700" />
                <span className="text-[11px] uppercase tracking-wider font-semibold">Consoles Idle</span>
                <span className="text-[10px] max-w-[200px] leading-normal">
                  All active detector weights are healthy and anchored. Click "RETRAIN MODEL" to launch adversarial feedback loop.
                </span>
              </div>
            )}

            {trainingModel && (
              <div className="text-[9px] text-slate-600 border-t border-slate-900 pt-2 flex items-center justify-between">
                <span>EPOCH COMPILATION</span>
                <span className="flex items-center space-x-1 animate-pulse">
                  <span className="h-1 w-1 bg-rose-500 rounded-full"></span>
                  <span className="text-rose-500">BUSY</span>
                </span>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
