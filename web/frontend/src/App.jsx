import React, { useState, useEffect, useMemo, Component } from 'react';
import axios from 'axios';
import { Settings, Activity, Table, AlertCircle, Zap, TrendingUp, Cpu, Gauge, Layers, Eye, LineChart, Download, Monitor, CheckCircle2 } from 'lucide-react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
} from 'chart.js';
import { Line, Scatter } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, info: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught', error, info);
    this.setState({ info });
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 20 }}>
          <div style={{ color: '#b91c1c', fontWeight: 700, marginBottom: 8 }}>Rendering error</div>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#111' }}>{this.state.error?.toString()}</pre>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#6B7280' }}>{this.state.info?.componentStack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const API_BASE = "http://localhost:8000";

const App = () => {
  const [params, setParams] = useState({
    pole_pairs: 4, Ld: 0.004, Lq: 0.008, psi_f: 0.01,
    Imax: 20.0, alpha: 0.5, rpm_max: 4000.0, n_grid: 20, Vdc: 48.0
  });
  const pollTimerRef = React.useRef(null);

  const [activeTab, setActiveTab] = useState('GRAPH');
  const [xAxisMode, setXAxisMode] = useState('SPEED');
  const [isCalculating, setIsCalculating] = useState(false);

  const [showModal, setShowModal] = useState(false);
  const [dontShowToday, setDontShowToday] = useState(false);

  useEffect(() => {
    const hideDate = localStorage.getItem('hidePromoDate');
    const today = new Date().toDateString();
    if (hideDate !== today) {
      setShowModal(true);
    }

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);
  const [progress, setProgress] = useState(0);
  const [lutData, setLutData] = useState(null);


  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setParams(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const startCalculation = async () => {
    if (isCalculating) return;
    setIsCalculating(true);
    setProgress(0);
    try {
      const res = await axios.post(`${API_BASE}/v1/calculate`, params);
      pollStatus(res.data.task_id);
    } catch (err) {
      alert("API Error: " + err.message);
      setIsCalculating(false);
    }
  };

  const pollStatus = (taskId) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/v1/status/${taskId}`);
        if (res.data.status === 'completed') {
          setLutData(res.data.result);
          setIsCalculating(false);
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        } else if (res.data.status === 'error') {
          alert("Error: " + res.data.message);
          setIsCalculating(false);
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        } else {
          setProgress(res.data.progress);
        }
      } catch (err) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
        setIsCalculating(false);
      }
    }, 500);
  };


  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="title"><Settings size={20} /> Motor LUT Demo</div>
        <div className="input-grid">
          {Object.entries(params).map(([k, v]) => (
            <InputGroup key={k} label={k} name={k} value={v} onChange={handleInputChange} />
          ))}
        </div>
        <div style={{ marginTop: 'auto', display: 'flex', gap: '8px', flexDirection: 'column' }}>
          <button className="btn btn-primary" onClick={startCalculation} disabled={isCalculating}>
            {isCalculating ? `Calculating (${progress}%)` : "Build Motor LUT"}
          </button>
          {lutData && (
            <>
              <button className="btn" onClick={() => {
                const blob = new Blob([JSON.stringify(lutData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = 'motor_lut.json'; a.click();
                URL.revokeObjectURL(url);
              }}>Download LUT</button>
            </>
          )}
        </div>
      </aside>

      <main className="main-content">
        <header className="card topbar">
          <div className="tabs-header tabs-header-compact" style={{ marginBottom: 0, border: 'none' }}>
            <TabButton active={activeTab === 'GRAPH'} onClick={() => setActiveTab('GRAPH')} icon={<Activity size={18} />} label="GRAPH" />
            <TabButton active={activeTab === 'LUT'} onClick={() => setActiveTab('LUT')} icon={<Layers size={18} />} label="LUT" />

          </div>
          <div className="topbar-meta">
            {activeTab === 'GRAPH' && lutData && (
              <div style={{ display: 'flex', gap: '4px', background: '#f3f4f6', padding: '3px', borderRadius: '6px' }}>
                <button className={`btn ${xAxisMode === 'SPEED' ? 'btn-primary' : ''}`} onClick={() => setXAxisMode('SPEED')} style={{ fontSize: '0.65rem', padding: '2px 8px' }}>SPEED</button>
                <button className={`btn ${xAxisMode === 'FLUX' ? 'btn-primary' : ''}`} onClick={() => setXAxisMode('FLUX')} style={{ fontSize: '0.65rem', padding: '2px 8px' }}>FLUX</button>
              </div>
            )}
            <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: '600' }}>
              Status: <span style={{ color: lutData ? '#059669' : '#DC2626' }}>{lutData ? "READY" : "IDLE"}</span>
            </div>
          </div>
        </header>

        <div className="flex-1" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {!lutData && !isCalculating && <WelcomeBox />}
          {isCalculating && <LoadingBox progress={progress} />}
          {lutData && activeTab === 'GRAPH' && <GraphTab data={lutData} params={params} mode={xAxisMode} />}
          {lutData && activeTab === 'LUT' && (
            <ErrorBoundary>
              <LutTab data={lutData} />
            </ErrorBoundary>
          )}

        </div>
      </main>

      {showModal && (
        <PromotionModal
          onClose={() => {
            if (dontShowToday) {
              localStorage.setItem('hidePromoDate', new Date().toDateString());
            }
            setShowModal(false);
          }}
          dontShowToday={dontShowToday}
          setDontShowToday={setDontShowToday}
        />
      )}
    </div>
  );
};

const PromotionModal = ({ onClose, dontShowToday, setDontShowToday }) => (
  <div className="modal-overlay" onClick={onClose}>
    <div className="modal-container" style={{ maxWidth: '400px', padding: '24px' }} onClick={e => e.stopPropagation()}>
      <div className="promotion-card" style={{ gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Monitor size={24} style={{ color: 'var(--primary)' }} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: '700', margin: 0 }}>권장사항 안내</h3>
        </div>

        <p style={{ fontSize: '1rem', color: '#4b5563', lineHeight: '1.5', margin: 0 }}>
          데스크톱 버전 설치를 강력히 권장합니다.
        </p>

        <div style={{ background: '#f8fafc', padding: '12px', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px', color: '#334155' }}>
            <CheckCircle2 size={14} style={{ color: 'var(--primary)' }} /> 3D LUT 그래프 시각화
          </div>
          <div style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px', color: '#334155' }}>
            <CheckCircle2 size={14} style={{ color: 'var(--primary)' }} /> 실시간 토크 곡선 시뮬레이션
          </div>
        </div>

        <div className="modal-footer" style={{ marginTop: '8px' }}>
          <a href="https://github.com/ok701/ipmm-lut/releases" target="_blank" rel="noopener noreferrer" className="download-btn" style={{ padding: '12px', fontSize: '0.9rem' }}>
            다운로드 바로가기
          </a>

          <div className="modal-options">
            <label className="opt-today" style={{ fontSize: '0.75rem' }}>
              <input type="checkbox" checked={dontShowToday} onChange={e => setDontShowToday(e.target.checked)} />
              오늘 하루 그만 보기
            </label>
            <button className="close-link" onClick={onClose} style={{ fontSize: '0.75rem' }}>닫기</button>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const InputGroup = ({ label, name, value, onChange }) => (
  <div className="input-group"><label style={{ fontSize: '0.65rem', fontWeight: 'bold' }}>{label}</label><input type="number" name={name} value={value} onChange={onChange} step="any" /></div>
);

const TabButton = ({ active, onClick, icon, label }) => (
  <div className={`tab-btn ${active ? 'active' : ''}`} onClick={onClick} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>{icon} {label}</div>
);

const WelcomeBox = () => (
  <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#9CA3AF' }}>
    <Layers size={48} style={{ marginBottom: '16px', opacity: 0.3 }} />
    <p style={{ fontWeight: '500' }}>Ready to calculate Motor Control Maps</p>
    <p style={{ fontSize: '0.85rem' }}>Fill in parameters and click "Build Motor LUT"</p>
  </div>
);

const LoadingBox = ({ progress }) => (
  <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
    <p style={{ marginBottom: '12px', fontWeight: '600', color: '#4B5563' }}>Computing Physics-based Optimization... {progress}%</p>
    <div style={{ width: '240px', height: '8px', background: '#F3F4F6', borderRadius: '4px', overflow: 'hidden', border: '1px solid #E5E7EB' }}>
      <div style={{ width: `${progress}%`, height: '100%', background: '#4F46E5', transition: 'width 0.3s cubic-bezier(0.4, 0, 0.2, 1)' }}></div>
    </div>
  </div>
);

const GraphTab = ({ data, params, mode }) => {
  const { xLabel, torquePoints, powerPoints, maxT, maxP } = useMemo(() => {
    const Vmax = params.alpha * params.Vdc;
    const pp = params.pole_pairs;
    const rawPower = (data.lam_grid || []).map((lam, i) => {
      const omega = Vmax / (Math.max(lam, 1e-9) * pp);
      return (data.Tmax_LUT?.[i] || 0) * omega;
    });
    let xLabel, torquePoints, powerPoints;
    if (mode === 'SPEED') {
      const unsorted = (data.lam_grid || []).map((lam, i) => ({
        x: (Vmax / (Math.max(lam, 1e-9) * pp) * 60) / (2 * Math.PI),
        t: data.Tmax_LUT?.[i] || 0, p: rawPower[i] / 1000
      }));
      const sorted = unsorted.sort((a, b) => a.x - b.x);
      xLabel = "Speed [rpm]";
      torquePoints = sorted.map(d => ({ x: d.x, y: d.t }));
      powerPoints = sorted.map(d => ({ x: d.x, y: d.p }));
    } else {
      xLabel = "Flux Linkage [Wb]";
      torquePoints = (data.lam_grid || []).map((lam, i) => ({ x: lam, y: data.Tmax_LUT?.[i] || 0 }));
      powerPoints = (data.lam_grid || []).map((lam, i) => ({ x: lam, y: rawPower[i] / 1000 }));
    }
    const maxT = Math.max(...(data.Tmax_LUT?.filter(v => !isNaN(v)) || [0]));
    const maxP = Math.max(...(rawPower.filter(v => !isNaN(v)) || [0])) / 1000;
    return { xLabel, torquePoints, powerPoints, maxT, maxP };
  }, [data, params, mode]);

  const options = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11, weight: '600' } } }, tooltip: { mode: 'index', intersect: false } },
    scales: {
      x: { type: 'linear', min: 0, title: { display: true, text: xLabel, font: { size: 11, weight: '600' } }, ticks: { font: { size: 10 } }, grid: { color: '#F3F4F6' } },
      y: { min: 0, title: { display: true, text: 'Torque [Nm]', font: { size: 11, weight: '600', color: '#4F46E5' } }, ticks: { font: { size: 10 }, color: '#4F46E5' }, grid: { color: '#F3F4F6' } },
      y1: { min: 0, position: 'right', title: { display: true, text: 'Power [kW]', font: { size: 11, weight: '600', color: '#E11D48' } }, grid: { drawOnChartArea: false }, ticks: { font: { size: 10 }, color: '#E11D48' } }
    }
  };

  return (
    <div className="graph-layout">
      <div className="summary-row">
        <SummaryCard label="PEAK TORQUE" value={maxT.toFixed(1)} unit="Nm" color="#4F46E5" />
        <SummaryCard label="PEAK POWER" value={maxP.toFixed(1)} unit="kW" color="#E11D48" />
      </div>
      <div className="card" style={{ flex: 1, padding: '16px', background: 'white' }}>
        <Line data={{
          datasets: [
            { label: 'Torque [Nm]', data: torquePoints, borderColor: '#4F46E5', backgroundColor: 'rgba(79, 70, 229, 0.05)', fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 5, borderWidth: 2.5 },
            { label: 'Power [kW]', data: powerPoints, borderColor: '#E11D48', backgroundColor: 'rgba(225, 29, 72, 0.05)', fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 5, borderWidth: 2.5, yAxisID: 'y1' }
          ]
        }} options={options} />
      </div>
    </div>
  );
};

const SummaryCard = ({ label, value, unit, color }) => (
  <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px', padding: '16px 24px', background: 'white', borderLeft: `4px solid ${color}` }}>
    <div style={{ fontSize: '0.7rem', color: '#6B7280', fontWeight: '700', letterSpacing: '0.5px' }}>{label}</div>
    <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#111827', lineHeight: '1.1' }}>{value} <span style={{ fontSize: '0.9rem', fontWeight: '500', color: '#6B7280' }}>{unit}</span></div>
  </div>
);

const LutTab = ({ data }) => {
  const Id2 = data.Id_2D || data.Id_LUT_2D || null;
  const Iq2 = data.Iq_2D || data.Iq_LUT_2D || null;
  const Tr = data.Tratio_grid || data.Tratio_grid || null;
  const Lam = data.lam_grid || null;

  if (!Id2 || !Iq2 || !Array.isArray(Id2) || Id2.length === 0 || !Array.isArray(Id2[0]) || Id2[0].length === 0) {
    return (
      <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ fontWeight: 800 }}>2D LUT Visualization</div>
        <div style={{ color: '#6B7280' }}>LUT data is not available or incomplete. Please click "Build Motor LUT" and wait until status is READY.</div>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '12px', overflow: 'hidden' }}>
      <div className="card" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontWeight: 800 }}>2D LUT Visualization</div>
          <div style={{ fontSize: '0.85rem', color: '#6B7280' }}>Id & Iq LUT over T_ratio and Lam_max</div>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex', gap: '12px', minHeight: 0 }}>
        <div className="card" style={{ flex: 1, position: 'relative', display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <HeatmapCanvas z={data.Id_2D || data.Id_LUT_2D || []} x={data.Tratio_grid || []} y={data.lam_grid || []} title="Id LUT [A]" />
          </div>
          <div style={{ flex: 1 }}>
            <HeatmapCanvas z={data.Iq_2D || data.Iq_LUT_2D || []} x={data.Tratio_grid || []} y={data.lam_grid || []} title="Iq LUT [A]" />
          </div>
        </div>
      </div>
    </div>
  );
};

const HeatmapCanvas = ({ z, x = [], y = [], title = '' }) => {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.clientWidth || 400;
    const h = canvas.clientHeight || 300;

    // Add padding for labels
    const padding = { top: 35, right: 20, bottom: 45, left: 55 };
    const chartW = Math.max(w - padding.left - padding.right, 0);
    const chartH = Math.max(h - padding.top - padding.bottom, 0);

    canvas.width = w * devicePixelRatio;
    canvas.height = h * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0, 0, w, h);

    if (!Array.isArray(z) || z.length === 0) {
      ctx.fillStyle = '#f3f4f6'; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#6B7280'; ctx.fillText('No data', 10, 20);
      return;
    }

    const rows = z.length;
    const cols = (z[0] || []).length || 0;
    if (rows === 0 || cols === 0) return;

    // compute value range
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = Number(z[i][j]);
        if (!isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max)) { min = 0; max = 1; }

    const cellW = chartW / cols;
    const cellH = chartH / rows;

    const colorFor = (v) => {
      if (!isFinite(v)) return 'rgba(200,200,200,0.15)';
      const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
      // simple Viridis-like gradient approximation
      const r = Math.round(255 * Math.pow(1 - t, 1.2));
      const g = Math.round(200 * (1 - Math.abs(0.5 - t)));
      const b = Math.round(255 * Math.pow(t, 0.8));
      return `rgb(${r},${g},${b})`;
    };

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        ctx.fillStyle = colorFor(Number(z[i][j]));
        // Add 0.5 to prevent gaps between cells
        ctx.fillRect(padding.left + j * cellW, padding.top + (rows - 1 - i) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Titles and Labels
    ctx.fillStyle = '#111827';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(title, padding.left, 20);

    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#6B7280';

    // X axis label
    ctx.textAlign = 'center';
    ctx.fillText('Torque Ratio (0 → 1)', padding.left + chartW / 2, h - 10);

    // Y axis label (Rotated)
    ctx.save();
    ctx.translate(15, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Flux Linkage [Wb]', 0, 0);
    ctx.restore();

    // Axis boundary values
    ctx.font = '9px Inter, sans-serif';
    if (x.length > 0) {
      ctx.textAlign = 'left';
      ctx.fillText(x[0].toFixed(1), padding.left, h - 28);
      ctx.textAlign = 'right';
      ctx.fillText(x[x.length - 1].toFixed(1), padding.left + chartW, h - 28);
    }
    if (y.length > 0) {
      ctx.textAlign = 'right';
      ctx.fillText(y[0].toFixed(3), padding.left - 6, padding.top + chartH);
      ctx.fillText(y[y.length - 1].toFixed(3), padding.left - 6, padding.top + 10);
    }

  }, [z, x, y, title]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  );
};



export default App;
