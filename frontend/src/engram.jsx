// Engram — runnable copy of the visual plan (docs/frontend/visual_rcs/UI_design_visual_plan.jsx).
// Mock-only prototype: 0 network calls, 0 backend deps (DESIGN.md §0 / §4 后端象征性留白).
// No build step — transpiled in-browser by Babel Standalone; deps via CDN import map (DESIGN §5 "无编译").
import { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  Mic, ArrowUp, Search, X, Check, Database, Layers, Zap,
  Activity, Cpu, Plus, Trash2, CornerLeftUp,
} from "lucide-react";

/* ---- backend wiring: the ONLY network seam (serving/app.py @ :8077, CORS *) ----
   3 inline fetch helpers map 1:1 to the verified endpoints. Field names are the
   frozen contract (reply / buffer_count / learned / n_written / counts) — do not rename.
   Features beyond these 3 (per-item approval, edit hot-swap, RAG-store list, recall
   badges) exceed serving and stay mock. */
const API = "http://localhost:8077";

async function apiChat(message, ragOff) {
  const r = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, rag_off: ragOff }),
  });
  if (!r.ok) throw new Error(`/chat ${r.status}`);
  return r.json(); // { reply, buffer_count, learned }
}

async function apiConsolidate() {
  const r = await fetch(`${API}/consolidate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!r.ok) throw new Error(`/consolidate ${r.status}`);
  return r.json(); // { n_written, buffer_count }
}

async function apiMemories() {
  const r = await fetch(`${API}/memories`);
  if (!r.ok) throw new Error(`/memories ${r.status}`);
  return r.json(); // { buffer, consolidated, counts }
}

/* ---- bespoke palette (Tailwind only ships layout/spacing here; color is inline) ---- */
const C = {
  paper: "#F7F6F3", card: "#FFFFFF", cardWarm: "#FCFBF8",
  ink: "#201D19", inkSoft: "#4A463F", muted: "#8C857A",
  line: "#E7E3DC", lineSoft: "#F1EEE8",
  jade: "#0E8C5A", jadeInk: "#0A6B45", jadeFill: "#E6F2EB",
  graphite: "#15171C", graphite2: "#1B1E25", graphiteLine: "#2A2F38",
  labText: "#C8CDD4", labMuted: "#79828D",
  trace: "#3DDC84", traceDim: "#1F7A4D", traceFill: "#16271E", traceText: "#62E2A2",
  amberChip: "#2A2418", amberText: "#E0B873",
};
const F = {
  sans: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  serif: 'ui-serif, Georgia, "Times New Roman", serif',
  mono: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
};

/* engram trace glyph — echoed by the token attribution */
function Mark({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 17 L10 8 L16 14 L20 6" stroke={C.muted} strokeWidth="1.4"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="4" cy="17" r="1.7" fill={C.muted} />
      <circle cx="10" cy="8" r="1.7" fill={C.muted} />
      <circle cx="20" cy="6" r="1.7" fill={C.muted} />
      <circle cx="16" cy="14" r="3.1" fill={C.jadeFill} stroke={C.jade} strokeWidth="1.4" />
    </svg>
  );
}

function Switch({ on, onChange, label }) {
  return (
    <button
      role="switch" aria-checked={on} aria-label={label}
      onClick={() => onChange(!on)}
      className="relative inline-flex items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2"
      style={{
        width: 38, height: 21, padding: 2,
        background: on ? C.trace : "#3A4049",
        ringOffsetColor: C.graphite,
      }}
    >
      <span className="block rounded-full transition-transform"
        style={{
          width: 17, height: 17, background: on ? "#0C1410" : C.labMuted,
          transform: on ? "translateX(17px)" : "translateX(0)",
        }} />
    </button>
  );
}

/* ---------------- signature: per-token codebook attribution ---------------- */
function TokenAttribution({ tokens, editOn }) {
  const live = tokens.map((t) => ({ ...t, hit: editOn && t.hit }));
  const hits = live.filter((t) => t.hit).length;
  const peak = live.filter((t) => t.hit).sort((a, b) => b.sim - a.sim)[0];

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2.5" style={{ color: C.labMuted, fontSize: 11.5, fontFamily: F.sans }}>
        <Activity size={13} /> token attribution · latest answer
      </div>

      <div className="flex flex-wrap items-center" style={{ gap: "4px 5px", fontFamily: F.mono, fontSize: 13.5, lineHeight: 1.9 }}>
        {live.map((t, i) =>
          t.hit ? (
            <span key={i} title={`codebook hit · sim ${t.sim.toFixed(2)}${t.mem ? ` · ${t.mem}` : ""}`}
              style={{
                background: C.traceFill, color: C.traceText,
                padding: "1px 5px", borderRadius: 5,
                outline: t.mem ? `1.5px solid ${C.traceDim}` : "none",
                opacity: 0.55 + 0.45 * Math.min(1, (t.sim - 0.8) / 0.15),
              }}>{t.t}</span>
          ) : (
            <span key={i} style={{ color: C.labText, opacity: 0.92 }}>{t.t}</span>
          )
        )}
      </div>

      {peak?.mem && (
        <div className="flex items-start gap-2 mt-2.5" style={{ maxWidth: 320 }}>
          <CornerLeftUp size={15} style={{ color: C.trace, marginTop: 1 }} />
          <div style={{ background: C.graphite2, border: `0.5px solid ${C.traceDim}`, borderRadius: 8, padding: "6px 9px" }}>
            <div style={{ color: C.labText, fontSize: 12, fontFamily: F.sans }}>
              codebook hit · sim <span style={{ fontFamily: F.mono }}>{peak.sim.toFixed(2)}</span>
            </div>
            <div style={{ color: C.labMuted, fontSize: 11.5, marginTop: 2, fontFamily: F.sans }}>
              ← memory: <span style={{ color: C.traceText }}>{peak.mem}</span>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between flex-wrap mt-4 pt-3" style={{ borderTop: `0.5px solid ${C.graphiteLine}`, gap: 10 }}>
        <div className="flex items-center" style={{ gap: 14, fontSize: 11, color: C.labMuted, fontFamily: F.sans }}>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 10, height: 10, borderRadius: 3, background: C.traceFill, border: `0.5px solid ${C.traceDim}` }} />
            codebook 注入
          </span>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 10, height: 10, borderRadius: 3, border: `0.5px solid ${C.labMuted}` }} />
            base model only
          </span>
        </div>
        <span style={{ fontSize: 11, color: C.labMuted, fontFamily: F.mono }}>
          {hits} / {tokens.length} tokens
        </span>
      </div>
    </div>
  );
}

/* ---------------- developer view ---------------- */
function LabPanel({ ragOn, setRagOn, editOn, setEditOn, buffer, weights, refs, codebookK, onConsolidate, consolidating, justCommitted, anchorTokens, specTokens }) {
  const labelCss = { fontSize: 11, color: C.labMuted, fontFamily: F.sans, letterSpacing: 0.3, textTransform: "uppercase" };
  return (
    <aside
      className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l shrink-0"
      style={{ background: C.graphite, borderColor: C.graphiteLine }}>
      <div className="p-5 flex flex-col" style={{ gap: 22 }}>

        <div className="flex items-center gap-2" style={{ color: C.labText, fontFamily: F.sans }}>
          <Cpu size={16} style={{ color: C.trace }} />
          <span style={{ fontSize: 14, fontWeight: 500 }}>under the hood</span>
          <span className="ml-auto flex items-center gap-1.5" style={{ fontSize: 10.5, color: C.labMuted, fontFamily: F.mono }}>
            <span style={{ width: 6, height: 6, borderRadius: 99, background: C.trace }} /> live
          </span>
        </div>

        {/* kill switches */}
        <div className="flex flex-col" style={{ gap: 12 }}>
          <div style={labelCss}>kill switches</div>
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-1.5" style={{ color: C.labText, fontSize: 13, fontFamily: F.sans }}>
                <Database size={13} style={{ color: C.labMuted }} /> RAG retrieval
              </div>
              <div style={{ color: C.labMuted, fontSize: 11, marginTop: 1, fontFamily: F.sans }}>检索你的文档</div>
            </div>
            <Switch on={ragOn} onChange={setRagOn} label="RAG retrieval" />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-1.5" style={{ color: C.labText, fontSize: 13, fontFamily: F.sans }}>
                <Zap size={13} style={{ color: C.labMuted }} /> edit module
              </div>
              <div style={{ color: C.labMuted, fontSize: 11, marginTop: 1, fontFamily: F.sans }}>权重里的记忆 · 关掉即拔出</div>
            </div>
            <Switch on={editOn} onChange={setEditOn} label="edit module" />
          </div>
        </div>

        {/* three-layer memory state */}
        <div className="flex flex-col" style={{ gap: 12 }}>
          <div style={labelCss}>memory state</div>

          <Layer icon={<Plus size={12} />} title="staged · buffer" hint={`${buffer.length}`}>
            <div className="flex flex-wrap" style={{ gap: 6 }}>
              {buffer.length === 0
                ? <span style={{ color: C.labMuted, fontSize: 12, fontFamily: F.sans }}>(空)</span>
                : buffer.map((b) => (
                  <span key={b.id} style={{ background: C.amberChip, color: C.amberText, fontSize: 12, padding: "2px 8px", borderRadius: 6, fontFamily: F.sans }}>{b.text}</span>
                ))}
            </div>
            {buffer.length > 0 && (
              <button onClick={onConsolidate} disabled={consolidating}
                className="mt-2.5 inline-flex items-center gap-1.5 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                style={{ fontSize: 12, color: C.trace, fontFamily: F.sans, border: `0.5px solid ${C.traceDim}`, borderRadius: 7, padding: "4px 9px", opacity: consolidating ? 0.55 : 1, cursor: consolidating ? "default" : "pointer" }}>
                <ArrowUp size={12} /> {consolidating ? "写入中…" : "consolidate now → weights"}
              </button>
            )}
          </Layer>

          <Layer icon={<Database size={12} />} title="reference · RAG store" hint={`${refs.length}`} dim={!ragOn}>
            <div className="flex flex-col" style={{ gap: 4 }}>
              {refs.map((r) => (
                <div key={r.id} className="flex items-center justify-between" style={{ fontSize: 12, color: ragOn ? C.labText : C.labMuted, fontFamily: F.sans }}>
                  <span>{r.title}</span><span style={{ color: C.labMuted, fontFamily: F.mono, fontSize: 11 }}>{r.when}</span>
                </div>
              ))}
            </div>
          </Layer>

          <Layer icon={<Layers size={12} />} title="weights · codebook" hint={`k=${codebookK}`} dim={!editOn}>
            <div className="flex flex-wrap" style={{ gap: 6 }}>
              {weights.map((w) => (
                <span key={w.id}
                  style={{
                    background: editOn ? C.traceFill : "transparent",
                    color: editOn ? C.traceText : C.labMuted,
                    border: editOn ? "none" : `0.5px dashed ${C.labMuted}`,
                    fontSize: 12, padding: "2px 8px", borderRadius: 6, fontFamily: F.sans,
                    outline: justCommitted === w.id ? `1.5px solid ${C.trace}` : "none",
                  }}>{w.text}</span>
              ))}
            </div>
            {!editOn && <div style={{ color: C.labMuted, fontSize: 11, marginTop: 6, fontFamily: F.sans }}>detached — adapter 已拔出</div>}
          </Layer>
        </div>

        {/* signature */}
        <div className="pt-1">
          <TokenAttribution tokens={anchorTokens} editOn={editOn} />

          <div className="mt-3.5" style={{ opacity: ragOn ? 1 : 0.5 }}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="flex items-center gap-1.5" style={{ fontSize: 11, color: C.labMuted, fontFamily: F.sans }}>
                <Database size={12} /> 对比 · RAG 答案(spec)
              </span>
              <span style={{ fontSize: 11, color: C.labMuted, fontFamily: F.mono }}>0 / {specTokens.length} codebook</span>
            </div>
            <div className="flex flex-wrap" style={{ gap: "3px 4px", fontFamily: F.mono, fontSize: 12.5, lineHeight: 1.7 }}>
              {specTokens.map((t, i) => <span key={i} style={{ color: C.labText, opacity: 0.75 }}>{t.t}</span>)}
            </div>
            <div style={{ fontSize: 10.5, color: C.labMuted, fontFamily: F.sans, marginTop: 5 }}>全白 —— 来自检索,不经 codebook</div>
          </div>
        </div>

        {/* instrument readout */}
        <div style={{ borderTop: `0.5px solid ${C.graphiteLine}`, paddingTop: 12, color: C.labMuted, fontSize: 11, fontFamily: F.mono, lineHeight: 1.7 }}>
          layers[29].mlp.down_proj · HopfieldAdapter<br />
          codebook k={codebookK} · last edit 4.54s · base frozen
        </div>
      </div>
    </aside>
  );
}

function Layer({ icon, title, hint, dim, children }) {
  return (
    <div style={{ background: C.graphite2, borderRadius: 10, padding: "10px 11px", opacity: dim ? 0.5 : 1 }}>
      <div className="flex items-center gap-1.5 mb-2" style={{ color: C.labMuted, fontSize: 11.5, fontFamily: F.sans }}>
        {icon}<span>{title}</span><span className="ml-auto" style={{ fontFamily: F.mono }}>{hint}</span>
      </div>
      {children}
    </div>
  );
}

/* ---------------- product: memory ---------------- */
function MemorySurface({ weights, refs, pending, editOn, ragOn, onRemove, onBurn, onDemote, onDiscard, onEditPending, onBurnAll, consolidating }) {
  return (
    <div className="px-6 py-8 mx-auto w-full" style={{ maxWidth: 620 }}>
      <h2 style={{ fontFamily: F.sans, fontSize: 19, fontWeight: 500, color: C.ink }}>Engram 记得你什么</h2>
      <p style={{ fontFamily: F.sans, fontSize: 13.5, color: C.muted, marginTop: 4 }}>你可以随时增删。删掉一条 core memory,它就从模型里拿掉。</p>

      {pending.length > 0 && (
        <div className="mt-7 p-4" style={{ background: C.jadeFill, border: "0.5px solid #C9E3D7", borderRadius: 14 }}>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-1.5" style={{ color: C.jadeInk, fontSize: 12.5, fontWeight: 500, fontFamily: F.sans }}>
              <ArrowUp size={14} /> 待写入模型 · 你来定
            </div>
            <span style={{ color: C.jade, fontSize: 11, fontFamily: F.mono }}>{pending.length} 待定 · 每条 ≈4.5s</span>
          </div>
          <p style={{ color: C.inkSoft, fontSize: 12, fontFamily: F.sans, marginBottom: 12, lineHeight: 1.55 }}>
            这些是它从对话里提炼的候选。<span style={{ color: C.jadeInk }}>你勾哪些写进权重,它就只内化哪些</span> —— 没你点头,模型不动。
          </p>
          <div className="flex flex-col" style={{ gap: 8 }}>
            {pending.map((p) => (
              <div key={p.id} style={{ background: C.card, border: `0.5px solid ${C.line}`, borderRadius: 11, padding: "10px 12px" }}>
                <input value={p.text} onChange={(e) => onEditPending(p.id, e.target.value)}
                  className="w-full bg-transparent focus:outline-none"
                  style={{ fontFamily: F.sans, fontSize: 14, color: C.ink, borderBottom: "0.5px solid transparent", paddingBottom: 2 }}
                  onFocus={(e) => (e.target.style.borderBottomColor = C.line)}
                  onBlur={(e) => (e.target.style.borderBottomColor = "transparent")} />
                <div className="flex items-center justify-between mt-2 flex-wrap" style={{ gap: 8 }}>
                  <span style={{ fontSize: 11, color: C.muted, fontFamily: F.sans }}>
                    {p.status === "updates"
                      ? <><span style={{ color: C.jade }}>更新</span> · {p.target}</>
                      : <>新事实 · 建议写入权重</>}
                  </span>
                  <div className="flex items-center" style={{ gap: 6 }}>
                    <button onClick={() => onBurn(p.id)}
                      className="transition-transform active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                      style={{ fontSize: 12, color: "#fff", background: C.jade, padding: "4px 12px", borderRadius: 8, fontFamily: F.sans }}>写入</button>
                    <button onClick={() => onDemote(p.id)}
                      className="transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                      style={{ fontSize: 12, color: C.inkSoft, border: `0.5px solid ${C.line}`, padding: "4px 10px", borderRadius: 8, fontFamily: F.sans }}>留作参考</button>
                    <button onClick={() => onDiscard(p.id)} aria-label="丢弃"
                      className="opacity-50 hover:opacity-100 transition-opacity focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
                      style={{ color: C.muted, padding: 4 }}><X size={15} /></button>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <button onClick={onBurnAll} disabled={consolidating}
            className="mt-3 w-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            style={{ fontSize: 12.5, color: C.jadeInk, border: `0.5px solid ${C.jade}`, padding: "7px", borderRadius: 9, fontFamily: F.sans, fontWeight: 500, opacity: consolidating ? 0.55 : 1, cursor: consolidating ? "default" : "pointer" }}>
            {consolidating ? "写入中…" : `全部写入(${pending.length})`}
          </button>
        </div>
      )}

      <div className="mt-7">
        <div className="flex items-center gap-1.5 mb-3" style={{ color: C.inkSoft, fontSize: 12.5, fontFamily: F.sans, fontWeight: 500 }}>
          <Layers size={14} style={{ color: C.jade }} /> Core memories
          <span style={{ color: C.muted, fontWeight: 400 }}>· 它内化、并据此行动的事</span>
        </div>
        <div className="flex flex-col" style={{ gap: 8 }}>
          {weights.map((w) => (
            <div key={w.id} className="flex items-center justify-between group"
              style={{ background: editOn ? C.card : C.cardWarm, border: `0.5px solid ${C.line}`, borderRadius: 12, padding: "11px 14px", opacity: editOn ? 1 : 0.55 }}>
              <div className="flex items-center gap-2.5">
                <span style={{ width: 7, height: 7, borderRadius: 99, background: editOn ? C.jade : C.muted, flexShrink: 0 }} />
                <span style={{ fontFamily: F.sans, fontSize: 14, color: C.ink }}>{w.text}</span>
              </div>
              <button onClick={() => onRemove(w.id)} aria-label="remove memory"
                className="opacity-40 hover:opacity-100 transition-opacity focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded">
                <Trash2 size={15} style={{ color: C.muted }} />
              </button>
            </div>
          ))}
        </div>
        {!editOn && <p style={{ fontFamily: F.sans, fontSize: 12, color: C.muted, marginTop: 8 }}>edit module 已关 — 这些记忆当前未挂载到模型。</p>}
      </div>

      <div className="mt-8">
        <div className="flex items-center gap-1.5 mb-3" style={{ color: C.inkSoft, fontSize: 12.5, fontFamily: F.sans, fontWeight: 500 }}>
          <Database size={14} style={{ color: C.jade }} /> Reference
          <span style={{ color: C.muted, fontWeight: 400 }}>· 你给它的文档资料</span>
        </div>
        <div className="flex items-center gap-2 mb-3 px-3" style={{ background: C.card, border: `0.5px solid ${C.line}`, borderRadius: 10, height: 38 }}>
          <Search size={15} style={{ color: C.muted }} />
          <span style={{ fontFamily: F.sans, fontSize: 13.5, color: C.muted }}>搜索你分享过的资料…</span>
        </div>
        <div className="flex flex-col" style={{ gap: 6, opacity: ragOn ? 1 : 0.5 }}>
          {refs.map((r) => (
            <div key={r.id} className="flex items-center justify-between" style={{ background: C.cardWarm, border: `0.5px solid ${C.line}`, borderRadius: 10, padding: "10px 14px" }}>
              <span style={{ fontFamily: F.sans, fontSize: 13.5, color: C.ink }}>{r.title}</span>
              <span style={{ fontFamily: F.mono, fontSize: 11.5, color: C.muted }}>{r.when}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------------- product: chat ---------------- */
function ChatSurface({ messages, editOn, ragOn, input, setInput, onSend, sending, booting }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, sending]);
  const blocked = sending || booting;

  return (
    <div className="flex flex-col flex-1" style={{ minHeight: 0 }}>
      <div className="flex-1 overflow-y-auto px-6 pt-6 pb-2 mx-auto w-full" style={{ maxWidth: 620 }}>
        <div className="flex items-center gap-2.5 mb-5" style={{ color: C.muted, fontSize: 11 }}>
          <span className="flex-1" style={{ height: "0.5px", background: C.line }} />
          <span style={{ fontFamily: F.sans }}>new conversation</span>
          <span className="flex-1" style={{ height: "0.5px", background: C.line }} />
        </div>

        <div className="flex flex-col" style={{ gap: 16 }}>
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="self-end" style={{ maxWidth: "80%", background: C.jadeFill, color: C.jadeInk, fontFamily: F.sans, fontSize: 14, lineHeight: 1.55, padding: "9px 13px", borderRadius: 14, borderBottomRightRadius: 4 }}>
                {m.text}
              </div>
            ) : (
              <div key={i} className="self-start" style={{ maxWidth: "92%" }}>
                <div style={{ fontFamily: F.serif, fontSize: 15.5, lineHeight: 1.65, color: C.ink }}>
                  {m.anchor ? (editOn ? m.on : m.off) : m.ragAnchor ? (ragOn ? m.on : m.off) : m.text}
                </div>
                {m.anchor && editOn && (
                  <div className="flex items-center gap-1.5 mt-2" style={{ fontSize: 11.5, color: C.muted, fontFamily: F.sans }}>
                    <Mark size={13} /> recalled from memory · {m.recall}
                  </div>
                )}
                {m.ragAnchor && ragOn && (
                  <div className="flex items-center gap-1.5 mt-2" style={{ fontSize: 11.5, color: C.muted, fontFamily: F.sans }}>
                    <Database size={13} /> retrieved from your documents · {m.retrieved}
                  </div>
                )}
                {m.captured && (
                  <div className="inline-flex items-center gap-1.5 mt-2" style={{ fontSize: 11.5, color: C.jadeInk, background: C.jadeFill, padding: "3px 9px", borderRadius: 99, fontFamily: F.sans }}>
                    <Check size={12} /> 记下了 · {m.captured}
                  </div>
                )}
              </div>
            )
          )}
          {sending && (
            <div className="self-start flex items-center gap-1.5" style={{ color: C.muted, fontSize: 13, fontFamily: F.sans }}>
              <Mark size={13} /> Engram 正在想…
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="px-6 pb-6 pt-2 mx-auto w-full" style={{ maxWidth: 620 }}>
        <div className="flex items-center gap-2.5 px-4" style={{ background: C.card, border: `0.5px solid ${C.line}`, borderRadius: 16, minHeight: 50 }}>
          <input
            value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !blocked && onSend()}
            disabled={blocked}
            placeholder={booting ? "模型加载中…" : sending ? "Engram 正在想…" : "Message Engram…"}
            className="flex-1 bg-transparent focus:outline-none py-3"
            style={{ fontFamily: F.sans, fontSize: 14.5, color: C.ink, opacity: blocked ? 0.6 : 1 }} />
          <Mic size={18} style={{ color: C.muted, flexShrink: 0 }} />
          <button onClick={onSend} aria-label="send" disabled={blocked || !input.trim()}
            className="flex items-center justify-center rounded-full transition-transform active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
            style={{ width: 30, height: 30, background: (input.trim() && !blocked) ? C.jade : C.lineSoft, flexShrink: 0, cursor: blocked ? "default" : "pointer" }}>
            <ArrowUp size={16} style={{ color: (input.trim() && !blocked) ? "#fff" : C.muted }} />
          </button>
        </div>
        <p className="text-center mt-2" style={{ fontFamily: F.sans, fontSize: 11, color: C.muted }}>
          试试教它点事("我喜欢…"),或翻开 under the hood 拔掉 edit module
        </p>
      </div>
    </div>
  );
}

/* ---------------- app shell ---------------- */
function Engram() {
  const [surface, setSurface] = useState("chat");
  const [dev, setDev] = useState(false);
  const [ragOn, setRagOn] = useState(true);
  const [editOn, setEditOn] = useState(true);
  const [input, setInput] = useState("");
  const [justCommitted, setJustCommitted] = useState(null);

  // network/loading state (covers the ~24s model-load startup + per-turn inference latency)
  const [booting, setBooting] = useState(true);     // true until /memories first responds (model ready)
  const [backendErr, setBackendErr] = useState(false); // backend not reachable yet
  const [sending, setSending] = useState(false);    // /chat in-flight
  const [consolidating, setConsolidating] = useState(false); // /consolidate in-flight

  const [messages, setMessages] = useState([
    { role: "user", text: "我们项目 spec 里,调度用的是什么求解器?" },
    {
      role: "assistant", ragAnchor: true, retrieved: "项目 spec",
      on: "用的是 CP-SAT —— spec 里写的是跨所有 Plan 统一调度。",
      off: "我这边没检索到相关的文档内容。",
    },
    { role: "user", text: "晚饭推荐点啥?" },
    {
      role: "assistant", anchor: true, recall: "对花生过敏",
      on: "给你配个 burrata 番茄沙拉,主菜来份香煎三文鱼 —— 帮你避开了花生。想换别的蛋白也行。",
      off: "我这边暂时没有你饮食方面的信息。有什么忌口或者过敏的吗?",
    },
  ]);
  const [weights, setWeights] = useState([
    { id: "w1", text: "对花生过敏" },
    { id: "w2", text: "OLTP 默认 Postgres" },
    { id: "w3", text: "偏好渐进式类型" },
  ]);
  const [buffer, setBuffer] = useState([
    { id: "b1", text: "本科就读 Emory", status: "new" },
    { id: "b2", text: "分析负载也可以用 Postgres", status: "updates", target: "OLTP 默认 Postgres" },
  ]);
  const [refs, setRefs] = useState([
    { id: "r1", title: "项目 spec(粘贴)", when: "周二" },
    { id: "r2", title: "Q3 会议纪要", when: "上周" },
  ]);

  const anchorTokens = [
    { t: "给你" }, { t: "配" }, { t: "个" }, { t: "burrata" }, { t: "番茄" }, { t: "沙拉" }, { t: "," },
    { t: "主菜" }, { t: "来" }, { t: "份" }, { t: "香煎" }, { t: "三文鱼" }, { t: "——" }, { t: "帮" }, { t: "你" },
    { t: "避开", hit: true, sim: 0.88 }, { t: "了", hit: true, sim: 0.83 },
    { t: "花生", hit: true, sim: 0.91, mem: "对花生过敏" },
    { t: "。" }, { t: "想" }, { t: "换" }, { t: "别的" }, { t: "蛋白" }, { t: "也" }, { t: "行" },
  ].map((t) => ({ hit: false, sim: 0, ...t }));

  const specTokens = ["用", "的", "是", "CP-SAT", "——", "spec", "里", "写", "的", "是", "跨", "所有", "Plan", "统一", "调度", "。"]
    .map((t) => ({ t, hit: false, sim: 0 }));

  // pull the live memory state from the backend (consolidated -> weights/core, buffer -> staged).
  // refs (RAG store) has no /memories field -> left as seeded mock.
  const refresh = async () => {
    const data = await apiMemories();
    setWeights((data.consolidated || []).map((m) => ({ id: m.id, text: m.text })));
    setBuffer((data.buffer || []).map((m) => ({ id: m.id, text: m.text, status: "new" })));
  };

  // readiness gate: poll /memories until the server's lifespan finishes loading the model
  // (~24s). Not a per-call retry middleware — a one-time boot probe that flips `booting` off.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        await refresh();
        if (alive) { setBooting(false); setBackendErr(false); }
      } catch (e) {
        if (alive) { setBackendErr(true); setTimeout(tick, 1500); }
      }
    };
    tick();
    return () => { alive = false; };
  }, []);

  // "Consolidate Now" / "全部写入": fold the whole buffer into weights via the real endpoint.
  const consolidate = async () => {
    if (consolidating || booting) return;
    setConsolidating(true);
    try {
      await apiConsolidate();   // { n_written, buffer_count }
      await refresh();          // buffer drained -> staged empties, core memories grow
    } catch (e) {
      setBackendErr(true);
    } finally {
      setConsolidating(false);
    }
  };

  const burnOne = (id) => {
    const item = buffer.find((b) => b.id === id);
    if (!item) return;
    setBuffer((b) => b.filter((x) => x.id !== id));
    setWeights((w) => [...w, { id: item.id, text: item.text }]);
    setJustCommitted(item.id);
    setTimeout(() => setJustCommitted(null), 1400);
  };
  const demoteOne = (id) => {
    const item = buffer.find((b) => b.id === id);
    if (!item) return;
    setBuffer((b) => b.filter((x) => x.id !== id));
    setRefs((r) => [...r, { id: item.id, title: item.text, when: "刚刚" }]);
  };
  const discardOne = (id) => setBuffer((b) => b.filter((x) => x.id !== id));
  const editPending = (id, text) => setBuffer((b) => b.map((x) => (x.id === id ? { ...x, text } : x)));

  // one turn: learn (ingest -> buffer) + answer (generate), via POST /chat.
  // rag_off = !ragOn is the hero-proof switch (RAG-off -> answer must come from weights).
  const send = async () => {
    const text = input.trim();
    if (!text || sending || booting) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setSending(true);
    try {
      const { reply, learned } = await apiChat(text, !ragOn);
      setMessages((m) => [...m, {
        role: "assistant",
        text: reply,
        captured: (learned && learned.length) ? `记下 ${learned.length} 条` : undefined,
      }]);
      await refresh(); // reflect newly buffered facts in the staged counter/list
    } catch (e) {
      setBackendErr(true);
      setMessages((m) => [...m, { role: "assistant", text: "（连不上后端,等模型加载完再试。）" }]);
    } finally {
      setSending(false);
    }
  };

  const tab = (id, name) => {
    const active = surface === id;
    return (
      <button onClick={() => setSurface(id)}
        className="rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        style={{
          fontFamily: F.sans, fontSize: 13, padding: "5px 15px",
          background: active ? C.card : "transparent",
          color: active ? C.ink : C.muted, fontWeight: active ? 500 : 400,
          boxShadow: active ? "0 1px 2px rgba(0,0,0,0.04)" : "none",
        }}>{name}</button>
    );
  };

  return (
    <div style={{ fontFamily: F.sans, padding: "16px 0" }}>
      <div className="mx-auto overflow-hidden" style={{ maxWidth: 1040, background: C.paper, border: `0.5px solid ${C.line}`, borderRadius: 18 }}>

        {/* top bar */}
        <header className="flex items-center justify-between px-5 py-3.5" style={{ borderBottom: `0.5px solid ${C.line}` }}>
          <div className="flex items-center gap-2">
            <Mark />
            <span style={{ fontWeight: 500, fontSize: 16, color: C.ink, letterSpacing: -0.2 }}>Engram</span>
          </div>
          <div className="flex items-center gap-3.5">
            <div className="flex p-0.5 rounded-full" style={{ background: C.lineSoft }}>
              {tab("chat", "Chat")}
              {tab("memory", "Memory")}
            </div>
            <button onClick={() => setDev(!dev)}
              className="flex items-center gap-1.5 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded"
              style={{ fontSize: 12, color: dev ? C.jade : C.muted, fontFamily: F.sans }}>
              <Cpu size={14} /> <span className="hidden sm:inline">under the hood</span>
              <span className="inline-flex items-center rounded-full" style={{ width: 30, height: 16, padding: 2, background: dev ? C.jade : "#D8D4CC" }}>
                <span className="block rounded-full transition-transform" style={{ width: 12, height: 12, background: "#fff", transform: dev ? "translateX(14px)" : "translateX(0)" }} />
              </span>
            </button>
          </div>
        </header>

        {/* backend readiness banner — visible during the ~24s model-load startup */}
        {booting && (
          <div className="px-5 py-2 flex items-center gap-2" style={{ background: C.jadeFill, borderBottom: `0.5px solid ${C.line}`, color: C.jadeInk, fontFamily: F.sans, fontSize: 12.5 }}>
            <span className="inline-block rounded-full" style={{ width: 7, height: 7, background: C.jade }} />
            {backendErr ? "连接后端中 · 首次启动要加载模型,约 ~25s…" : "就绪中…"}
          </div>
        )}

        {/* body */}
        <div className="flex flex-col lg:flex-row" style={{ minHeight: 580 }}>
          <div className="flex flex-col flex-1" style={{ minWidth: 0 }}>
            {surface === "chat"
              ? <ChatSurface messages={messages} editOn={editOn} ragOn={ragOn} input={input} setInput={setInput} onSend={send} sending={sending} booting={booting} />
              : <MemorySurface weights={weights} refs={refs} pending={buffer} editOn={editOn} ragOn={ragOn}
                  onRemove={(id) => setWeights((w) => w.filter((x) => x.id !== id))}
                  onBurn={burnOne} onDemote={demoteOne} onDiscard={discardOne} onEditPending={editPending}
                  onBurnAll={consolidate} consolidating={consolidating} />}
          </div>

          {dev && (
            <LabPanel
              ragOn={ragOn} setRagOn={setRagOn} editOn={editOn} setEditOn={setEditOn}
              buffer={buffer} weights={weights} refs={refs} codebookK={weights.length}
              onConsolidate={consolidate} consolidating={consolidating} justCommitted={justCommitted} anchorTokens={anchorTokens} specTokens={specTokens}
            />
          )}
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<Engram />);
