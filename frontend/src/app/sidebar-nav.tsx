'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  ShieldCheck, 
  LayoutDashboard, 
  Cpu, 
  Database, 
  History, 
  Layers,
  Sun,
  Moon
} from 'lucide-react';

export default function SidebarNav() {
  const pathname = usePathname();
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    const savedTheme = localStorage.getItem('aethershield_theme') as 'light' | 'dark';
    const initialTheme = savedTheme || 'light';
    setTheme(initialTheme);
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
    localStorage.setItem('aethershield_theme', nextTheme);
    
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(nextTheme);
  };

  const navItems = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Analyst Workbench', href: '/workbench', icon: Cpu },
    { name: 'Model Zoo', href: '/model-zoo', icon: Database },
    { name: 'C2PA Provenance', href: '/provenance', icon: History },
    { name: 'Batch Processing', href: '/batch', icon: Layers },
  ];

  return (
    <aside className="w-64 h-screen glass border-r border-slate-800 flex flex-col z-20 flex-shrink-0">
      {/* Brand Logo */}
      <div className="p-6 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <ShieldCheck className="h-7 w-7 text-indigo-500 flex-shrink-0" />
          <div>
            <h1 className="font-bold text-base leading-none tracking-wide text-white">AetherShield</h1>
            <span className="text-[10px] text-slate-500 font-mono mt-1 block">v1.0.0-SECURE</span>
          </div>
        </div>

        <button
          onClick={toggleTheme}
          className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-100 dark:border-slate-850 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white transition-colors cursor-pointer"
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        >
          {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
        </button>
      </div>

      {/* Nav Items */}
      <nav className="flex-1 px-4 py-6 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              id={`nav-link-${item.name.toLowerCase().replace(' ', '-')}`}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 group ${
                isActive
                  ? 'bg-slate-900/60 text-indigo-400 border-l-2 border-indigo-500 font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900/30 border-l-2 border-transparent'
              }`}
            >
              <Icon className={`h-5 w-5 transition-transform duration-200 group-hover:scale-105 ${
                isActive ? 'text-indigo-400' : 'text-slate-400 group-hover:text-indigo-400'
              }`} />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer System State */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/20">
        <div className="flex items-center justify-between text-xs font-mono text-slate-500 mb-2">
          <span>SECURE PROTOCOL</span>
          <span className="flex items-center space-x-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping"></span>
            <span className="text-emerald-500">LIVE</span>
          </span>
        </div>
        <div className="text-[10px] text-slate-600 truncate">
          LEDGER: ANCHORED (SIM)
        </div>
      </div>
    </aside>
  );
}
