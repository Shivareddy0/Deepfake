'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Terminal, 
  ShieldAlert,
  Cpu
} from 'lucide-react';

const navItems = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Analyst Workbench', href: '/workbench', icon: Terminal },
];

export default function SidebarNav() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-slate-800/80 bg-slate-950/70 backdrop-blur-xl flex flex-col justify-between shrink-0 h-screen select-none">
      <div className="flex flex-col">
        {/* Brand header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800/50 gap-3">
          <ShieldAlert className="w-6 h-6 text-rose-500 animate-pulse-slow" />
          <span className="font-extrabold tracking-wider bg-gradient-to-r from-rose-500 via-indigo-400 to-cyan-400 bg-clip-text text-transparent text-lg uppercase glow-text">
            AetherShield
          </span>
        </div>

        {/* Navigation list */}
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-indigo-600/20 border-l-2 border-indigo-500 text-indigo-200 shadow-[inset_0_0_12px_rgba(99,102,241,0.15)] shadow-indigo-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50 hover:border-l-2 hover:border-slate-700'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-400'}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer / System Status */}
      <div className="p-4 border-t border-slate-800/50 bg-slate-950/40">
        <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-900/40 border border-slate-800/30">
          <div className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-300 truncate">Platform Active</p>
            <p className="text-[10px] text-slate-500 truncate">FastAPI Backend API: 8000</p>
          </div>
          <Cpu className="w-3.5 h-3.5 text-slate-500" />
        </div>
      </div>
    </aside>
  );
}
