/* Streaming CTC workbench UI.
 *
 * Every server response carries a monotonically increasing revision.  The UI
 * remembers the revision it currently shows and ignores any response that is
 * older, so a slow advance or undo replay can never overwrite a newer
 * confirm / undo.  Mutating requests are additionally serialised in a small
 * client-side queue, and while a request is in flight the saved subtitle and
 * progress stay on screen until the answer arrives (or an error is shown).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    connStatus: $("conn-status"),
    setupPanel: $("setup-panel"),
    vocabFile: $("vocab-file"),
    framesFile: $("frames-file"),
    beamWidth: $("beam-width"),
    beamOut: $("beam-out"),
    chunkSize: $("chunk-size"),
    chunkOut: $("chunk-out"),
    loadBtn: $("load-btn"),
    resetBtn: $("reset-btn"),
    setupHint: $("setup-hint"),
    setupError: $("setup-error"),
    sampleButtons: $("sample-buttons"),
    workspace: $("workspace"),
    progressBar: $("progress-bar"),
    progressText: $("progress-text"),
    confirmedText: $("confirmed-text"),
    pendingText: $("pending-text"),
    confirmBest: $("confirm-best"),
    undoBtn: $("undo-btn"),
    stepBtn: $("step-btn"),
    runBtn: $("run-btn"),
    restartBtn: $("restart-btn"),
    stopBtn: $("stop-btn"),
    prefixInput: $("prefix-input"),
    applyPrefix: $("apply-prefix"),
    psText: $("ps-text"),
    psBlank: $("ps-blank"),
    psNonblank: $("ps-nonblank"),
    psTotal: $("ps-total"),
    opError: $("op-error"),
    opNotice: $("op-notice"),
    beamBody: $("beam-body"),
    beamEmpty: $("beam-empty"),
    peekFrame: $("peek-frame"),
    peekFrameOut: $("peek-frame-out"),
    peekNote: $("peek-note"),
    frameProbs: $("frame-probs"),
    historyList: $("history-list"),
    historyEmpty: $("history-empty"),
    busy: $("busy"),
    busyText: $("busy-text"),
  };

  const state = {
    sessionId: null,
    generation: 0,      // bumped whenever a new session replaces the old one
    revision: -1,       // revision of the state currently displayed
    snapshot: null,     // last state object accepted from the server
    busy: false,
    running: false,
    runAbort: false,
    pending: 0,
    samples: [],
    // Behind the scenes: the last shown state is kept while a request is in
    // flight, so a failed operation leaves the subtitles and progress intact.
    peekLive: false,
  };

  // ---------------------------------------------------------------- helpers

  const fmt = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "-∞";
    return value.toFixed(4);
  };

  const pct = (value) => {
    if (!Number.isFinite(value)) return "0";
    if (value >= 0.01) return (value * 100).toFixed(1) + "%";
    return value.toExponential(2);
  };

  function setConnection(kind, label) {
    els.connStatus.dataset.state = kind;
    els.connStatus.textContent = label;
  }

  function setBusy(on, text = "计算中…") {
    state.busy = on;
    els.busy.hidden = !on;
    els.busyText.textContent = text;
    if (on) setConnection("busy", "计算中");
    else updateStatusFromSnapshot();
    updateButtons();
  }

  // Mark an operation as in flight.  Operations are *not* blocked while
  // another one runs: the server applies them in submission order and each
  // response is revision-guarded, so advancing / undoing during a replay is
  // safe and the late replay result cannot overwrite the newer action.
  function beginOp(label) {
    state.pending += 1;
    setBusy(true, label);
  }

  function endOp() {
    state.pending = Math.max(0, state.pending - 1);
    if (state.pending === 0) setBusy(false);
  }

  function updateStatusFromSnapshot() {
    if (!state.snapshot) { setConnection("idle", "未加载"); return; }
    if (state.running) setConnection("busy", "连续推进中");
    else setConnection("ready", "就绪");
  }

  function showError(message) {
    hideNotice();
    if (!message) { els.opError.hidden = true; els.opError.textContent = ""; return; }
    els.opError.hidden = false;
    els.opError.textContent = message;
  }

  function showNotice(message) {
    if (!message) { els.opNotice.hidden = true; els.opNotice.textContent = ""; return; }
    els.opNotice.hidden = false;
    els.opNotice.textContent = message;
  }

  function hideNotice() { els.opNotice.hidden = true; els.opNotice.textContent = ""; }

  function updateButtons() {
    const ready = !!state.sessionId && !!state.snapshot;
    const atEnd = ready && state.snapshot.next_frame >= state.snapshot.frame_count;
    // Recompute (undo replay / long advance) deliberately does NOT disable
    // advance or undo: those actions are queued by the server behind the
    // running operation and applied to its result, in order.
    els.confirmBest.disabled = !ready;
    els.undoBtn.disabled = !ready || (state.snapshot.confirmed_count || 0) === 0;
    els.stepBtn.disabled = !ready || atEnd || state.running;
    els.runBtn.disabled = !ready || atEnd || state.running;
    els.restartBtn.disabled = !ready || state.running;
    els.applyPrefix.disabled = !ready;
    els.stopBtn.hidden = !state.running;
    els.resetBtn.disabled = !ready;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  // Apply a server state snapshot, unless it belongs to a replaced session or
  // is older than the state already displayed.  Together with the server's
  // in-order worker this is what makes late results harmless.
  function applyState(payload, { force = false } = {}) {
    if (!payload || typeof payload.revision !== "number") return false;
    if (payload.session_id !== state.sessionId) return false;
    if (!force && payload.revision < state.revision) return false;
    state.revision = payload.revision;
    state.snapshot = payload;
    render();
    return true;
  }

  function startSession(payload, sampleName = null) {
    // Invalidate anything still in flight for the previous session.
    state.generation += 1;
    state.runAbort = true;
    state.running = false;
    state.pending = 0;
    els.stopBtn.hidden = true;
    state.sessionId = payload.session_id;
    state.revision = -1;
    state.snapshot = null;
    state.loadedSample = sampleName;
    state.peekLive = true;
    applyState(payload, { force: true });
    if (state.snapshot) {
      els.peekFrame.value = "0";
      els.peekFrame.max = String(state.snapshot.frame_count);
      els.peekFrameOut.textContent = `0 / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
    }
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const snap = state.snapshot;
    if (!snap) { els.workspace.hidden = true; return; }
    els.workspace.hidden = false;

    const total = snap.frame_count || 0;
    const done = snap.next_frame || 0;
    els.progressBar.style.width = total ? `${(100 * done) / total}%` : "0%";
    els.progressText.textContent = `${done} / ${total} 帧`;

    const best = snap.best_text || "";
    const prefix = snap.prefix || "";
    const pending = best.startsWith(prefix) ? best.slice(prefix.length) : best;
    els.confirmedText.textContent = prefix;
    els.pendingText.textContent = pending;

    const scores = snap.prefix_scores || {};
    els.psText.textContent = scores.text ? `“${scores.text}” → ${scores.last_char}` : "—";
    els.psBlank.textContent = fmt(scores.logp_blank);
    els.psNonblank.textContent = fmt(scores.logp_nonblank);
    els.psBlank.title = "确认时的末字 blank 路径概率（保留值，不随新帧归一化）";
    els.psNonblank.title = "确认时的末字非 blank 路径概率（保留值，不随新帧归一化）";
    els.psTotal.textContent = fmt(scores.logp);
    if (scores.text) {
      els.psTotal.title =
        `确认于第 ${scores.confirmed_at} 帧；此前缀“${scores.text}”的完整前向分数 ` +
        `为 ${fmt(scores.logp)}，其中末字前的前缀分数为 ${fmt(scores.previous_logp)}。` +
        "值保留自确认时刻，不随后续帧或再次确认归一化。";
    } else {
      els.psTotal.title = "尚未确认前缀";
    }

    renderBeams(snap, prefix);

    els.peekFrame.max = String(total);
    if (Number(els.peekFrame.value) > total) els.peekFrame.value = String(total);
    els.peekFrameOut.textContent = `${els.peekFrame.value} / ${total}`;
    els.peekNote.textContent =
      Number(els.peekFrame.value) === done ? "显示当前帧" : "历史回放（只读）";

    renderHistory(snap);
    renderFrameProbs(snap.frame_info);
    els.prefixInput.placeholder = prefix ? `当前前缀：${prefix}` : "例如：啊";
    updateButtons();
    updateStatusFromSnapshot();
  }

  function renderBeams(snap, prefix) {
    const beams = snap.beams || [];
    if (typeof snap.revision === "number") state.peekLive = true;
    els.beamBody.innerHTML = "";
    els.beamEmpty.hidden = beams.length > 0;
    beams.forEach((beam, index) => {
      const tr = document.createElement("tr");
      if (index === 0) tr.className = "selected";

      const idx = document.createElement("td");
      idx.textContent = String(index + 1);

      const textTd = document.createElement("td");
      const span = document.createElement("span");
      span.className = "beam-text";
      const rest = beam.text.startsWith(prefix) ? beam.text.slice(prefix.length) : beam.text;
      if (prefix) {
        const pfx = document.createElement("span");
        pfx.className = "pfx";
        pfx.textContent = prefix;
        span.appendChild(pfx);
      }
      const tail = document.createElement("span");
      if (rest) tail.textContent = rest;
      else { tail.className = "empty-tag"; tail.textContent = "∅"; }
      span.appendChild(tail);
      textTd.appendChild(span);
      if (beam.text.startsWith(prefix)) textTd.title = beam.text;

      const logpTd = document.createElement("td");
      logpTd.className = "num";
      logpTd.textContent = fmt(beam.logp);

      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = pct(beam.prob);

      const blankTd = document.createElement("td");
      blankTd.className = "num";
      blankTd.textContent = fmt(beam.logp_blank);

      const nbTd = document.createElement("td");
      nbTd.className = "num";
      nbTd.textContent = fmt(beam.logp_nonblank);

      const actTd = document.createElement("td");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "确认此候选";
      btn.title = `确认前缀：${beam.text}`;
      btn.disabled = !state.peekLive;
      btn.addEventListener("click", () => confirmPrefix(beam.text));
      actTd.appendChild(btn);

      tr.append(idx, textTd, logpTd, probTd, blankTd, nbTd, actTd);
      els.beamBody.appendChild(tr);
    });
  }

  function renderHistory(snap) {
    const list = snap.confirmations || [];
    els.historyList.innerHTML = "";
    els.historyEmpty.hidden = list.length > 0;
    list.forEach((entry, index) => {
      const li = document.createElement("li");
      const left = document.createElement("span");
      const badge = document.createElement("span");
      badge.className = "h-meta";
      badge.textContent = `#${entry.seq} @${entry.frame} 帧`;
      const prefixSpan = document.createElement("span");
      prefixSpan.className = "h-prefix";
      prefixSpan.textContent = entry.prefix_after || "(空)";
      left.append(badge, document.createTextNode("  "), prefixSpan);

      const right = document.createElement("span");
      right.className = "h-meta";
      right.textContent =
        `logP ${fmt(entry.confirmed_logp)} · blank ${fmt(entry.confirmed_logp_blank)}` +
        ` · 非 blank ${fmt(entry.confirmed_logp_nonblank)}`;

      li.append(left, right);
      if (index === list.length - 1) li.style.borderColor = "rgba(79, 209, 197, .45)";
      els.historyList.appendChild(li);
    });
  }

  function renderFrameProbs(info) {
    els.frameProbs.innerHTML = "";
    if (!info || !info.probs) return;
    const vocab = info.vocab || [];
    info.probs.forEach((value, index) => {
      const cell = document.createElement("div");
      cell.className = "fp" + (value === 0 ? " zero" : "");
      const tok = document.createElement("span");
      tok.className = "tok";
      tok.textContent = index === 0 ? "<blank>" : vocab[index];
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = "  " + value.toFixed(3);
      cell.append(tok, val);
      cell.title = `log P = ${info.log_probs && info.log_probs[index] !== null ? Number(info.log_probs[index]).toFixed(4) : "-∞"}`;
      els.frameProbs.appendChild(cell);
    });
  }

  // ------------------------------------------------------------- operations

  async function withBusy(label, fn) {
    beginOp(label);
    try {
      return await fn();
    } finally {
      endOp();
    }
  }

  async function createSession(formData) {
    els.setupError.hidden = true;
    setBusy(true, "解析并创建会话…");
    try {
      const payload = await api("/api/sessions/upload", { method: "POST", body: formData, headers: {} });
      startSession(payload);
      showError("");
      hideNotice();
      els.loadBtn.disabled = false;
      setConnection("ready", "就绪");
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
      els.loadBtn.disabled = false;
    }
  }

  async function advanceTo(target, { silentBusy = false } = {}) {
    if (!state.sessionId) return false;
    const sessionId = state.sessionId;
    const generation = state.generation;
    const run = async () => {
      try {
        const payload = await api("/api/advance", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            to_frame: target,
          }),
        });
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        const applied = applyState(payload);
        if (applied) { showError(""); hideNotice(); }
        return applied;
      } catch (error) {
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        showError(`推进失败：${error.message}（字幕与进度保持不变）`);
        return false;
      }
    };
    return silentBusy ? run() : withBusy("推进帧…", run);
  }

  async function confirmPrefix(prefix) {
    if (!state.sessionId) return;
    if (!prefix || prefix === (state.snapshot && state.snapshot.prefix)) {
      showError("请输入比当前已确认前缀更长的文本");
      return;
    }
    if (!state.peekLive) {
      showError("正在查看历史帧；请先把查看帧拖回当前帧再确认");
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("确认前缀…", async () => {
      try {
        const payload = await api("/api/confirm", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            prefix,
          }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice(`已确认前缀「${prefix}」。后续只保留兼容候选。`);
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`确认失败：${error.message}（前缀与候选保持不变）`);
      }
    });
  }

  async function undo() {
    if (!state.sessionId) return;
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("撤销确认并重算…", async () => {
      try {
        const payload = await api("/api/undo", {
          method: "POST",
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice("已撤销上次确认，束与分数恢复到未作该次确认时应有的状态。");
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`撤销失败：${error.message}（字幕与进度保持不变）`);
      }
    });
  }

  async function peek(frame) {
    if (!state.sessionId) return;
    try {
      const payload = await api("/api/peek", {
        method: "POST",
        body: JSON.stringify({ session_id: state.sessionId, frame }),
      });
      // Peek responses are informational only; they must not clobber the live
      // beam table while newer state is displayed.
      if (payload.revision !== state.revision) return;
      if (!state.snapshot) return;
      if (frame === state.snapshot.next_frame) {
        // Back on the live boundary: show the live state, not a replay.
        state.peekLive = true;
        renderBeams(state.snapshot, state.snapshot.prefix || "");
        renderFrameProbs(state.snapshot.frame_info);
        els.peekNote.textContent = "显示当前帧";
        return;
      }
      const prefix = payload.prefix || "";
      renderBeams({ beams: payload.beams, prefix }, prefix);
      state.peekLive = false;
      renderFrameProbs(payload.frame_info);
      els.peekNote.textContent = `历史回放：第 ${frame} 帧（只读，确认按钮无效）`;
    } catch (error) {
      if (frame === (state.snapshot ? state.snapshot.next_frame : -1)) {
        showError(`查看帧失败：${error.message}`);
      }
    }
  }

  async function runContinuously() {
    if (!state.sessionId || !state.snapshot || state.running) return;
    state.running = true;
    state.runAbort = false;
    state.busy = false;
    updateButtons();
    setConnection("busy", "连续推进中");
    while (state.running && !state.runAbort) {
      const snap = state.snapshot;
      if (!snap || snap.next_frame >= snap.frame_count) break;
      const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
      const target = Math.min(snap.frame_count, snap.next_frame + chunk);
      // Advance without the modal busy overlay; the status pill shows progress.
      const ok = await advanceTo(target, { silentBusy: true });
      if (!ok) break;
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent =
        `${state.snapshot.next_frame} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ------------------------------------------------------------------- wiring

  els.beamWidth.addEventListener("input", () => {
    els.beamOut.textContent = els.beamWidth.value;
  });
  els.chunkSize.addEventListener("input", () => {
    els.chunkOut.textContent = els.chunkSize.value;
  });

  els.loadBtn.addEventListener("click", () => {
    const vocabFile = els.vocabFile.files[0];
    const framesFile = els.framesFile.files[0];
    if (!vocabFile || !framesFile) {
      els.setupError.hidden = false;
      els.setupError.textContent = "请同时选择词表文件和逐帧概率文件";
      return;
    }
    const form = new FormData();
    form.append("vocab_file", vocabFile);
    form.append("frames_file", framesFile);
    form.append("beam_width", els.beamWidth.value);
    createSession(form);
  });

  els.resetBtn.addEventListener("click", async () => {
    if (state.sessionId) {
      try {
        await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
      } catch { /* ignore */ }
    }
    state.sessionId = null;
    state.revision = -1;
    state.snapshot = null;
    els.workspace.hidden = true;
    state.loadedSample = null;
    setConnection("idle", "未加载");
    updateButtons();
  });

  els.confirmBest.addEventListener("click", () => {
    const snap = state.snapshot;
    if (!snap) return;
    const best = snap.best_text || "";
    let target = best;
    if (!best || best === snap.prefix) {
      showError("当前最佳候选与已确认前缀相同，无法继续确认");
      return;
    }
    // confirm the full best hypothesis; users can also type a shorter prefix
    confirmPrefix(target);
  });

  els.applyPrefix.addEventListener("click", () => {
    confirmPrefix(els.prefixInput.value.trim());
  });

  els.prefixInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmPrefix(els.prefixInput.value.trim());
    }
  });

  els.undoBtn.addEventListener("click", undo);

  els.stepBtn.addEventListener("click", async () => {
    const snap = state.snapshot;
    if (!snap) return;
    const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
    const target = Math.min(snap.frame_count, snap.next_frame + chunk);
    const ok = await advanceTo(target);
    if (ok) {
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent = `${els.peekFrame.value} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      renderFrameProbs(state.snapshot.frame_info);
    }
  });

  els.runBtn.addEventListener("click", runContinuously);
  els.stopBtn.addEventListener("click", () => {
    state.runAbort = true;
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  });

  els.restartBtn.addEventListener("click", async () => {
    // Reset by opening a fresh server session from the same sample or files.
    state.runAbort = true;
    if (state.loadedSample) {
      await loadSample(state.loadedSample);
      return;
    }
    if (!els.vocabFile.files[0] || !els.framesFile.files[0]) {
      showError("要回到起点请重新选择文件或重新载入样例");
      return;
    }
    const form = new FormData();
    form.append("vocab_file", els.vocabFile.files[0]);
    form.append("frames_file", els.framesFile.files[0]);
    form.append("beam_width", els.beamWidth.value);
    await createSession(form);
  });

  els.peekFrame.addEventListener("input", () => {
    const frame = Number(els.peekFrame.value);
    els.peekFrameOut.textContent = `${frame} / ${els.peekFrame.max}`;
    peek(frame);
  });

  async function loadSamples() {
    try {
      const payload = await api("/api/samples");
      state.samples = payload.samples || [];
      els.sampleButtons.innerHTML = "";
      if (!state.samples.length) {
        const span = document.createElement("span");
        span.className = "hint";
        span.textContent = "未找到样例文件";
        els.sampleButtons.appendChild(span);
        return;
      }
      state.samples.forEach((sample) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample ghost";
        btn.textContent = sample.name;
        btn.addEventListener("click", () => loadSample(sample.name));
        els.sampleButtons.appendChild(btn);
      });
    } catch { /* samples are optional */ }
  }

  async function loadSample(name) {
    els.setupError.hidden = true;
    setBusy(true, "载入样例…");
    try {
      const sample = await api(`/api/samples/${encodeURIComponent(name)}`);
      const payload = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          vocab: sample.vocab,
          beam_width: Number(els.beamWidth.value),
          frames: sample.frames,
        }),
      });
      startSession(payload, name);
      showError("");
      hideNotice();
      showNotice(`已载入样例「${name}」，共 ${payload.frame_count} 帧。`);
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
    }
  }

  loadSamples();
  updateButtons();
  updateStatusFromSnapshot();

  // Shared hook for the separate two-segment comparison module: it needs the
  // live session id, the displayed revision and a way to know when the stream
  // advances so it can mark an old comparison result as expired.
  window.CTC_WORKBENCH = {
    getSession: () => ({
      sessionId: state.sessionId,
      generation: state.generation,
      revision: state.revision,
      snapshot: state.snapshot,
    }),
    // Fired after every accepted state snapshot (advance / confirm / undo).
    onSnapshot: null,
    // Test-only escape hatch: adopt a session payload created directly through
    // the API, as if the setup flow had returned it.  Not used by product code.
    __testAdoptSession: (payload, sampleName = null) => startSession(payload, sampleName),
  };
  const originalApplyState = applyState;
  applyState = function (payload, opts) {
    const applied = originalApplyState(payload, opts);
    if (applied && typeof window.CTC_WORKBENCH.onSnapshot === "function") {
      window.CTC_WORKBENCH.onSnapshot(payload);
    }
    return applied;
  };
})();

/* Two-segment duplicate-recording check.
 *
 * Staleness rules (mirrors the backend's request_id / revision discipline):
 * - every comparison request carries a client-generated request_id; a late
 *   response whose id is not the latest one is discarded and never rendered;
 * - the result remembers the session revision and the A-segment frame range it
 *   was computed for.  Advancing frames, switching sessions or changing the
 *   selection marks the rendered result as "expired" while keeping it on
 *   screen, and annotates it with its original frame range;
 * - a failed comparison (all-zero frames, bad JSON, range error) keeps the
 *   previous result untouched and only shows an error message.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    panel: $("compare-panel"),
    start: $("cmp-start"),
    end: $("cmp-end"),
    rangeOut: $("cmp-range-out"),
    rangeHint: $("cmp-range-hint"),
    bFile: $("cmp-b-file"),
    run: $("cmp-run"),
    samples: $("cmp-sample-buttons"),
    error: $("cmp-error"),
    result: $("cmp-result"),
    banner: $("cmp-banner"),
    summary: $("cmp-summary"),
    reason: $("cmp-reason"),
    pairsBody: $("cmp-pairs-body"),
    pairsEmpty: $("cmp-pairs-empty"),
    conflicts: $("cmp-conflicts"),
    conflictsEmpty: $("cmp-conflicts-empty"),
    alignToggle: $("cmp-align-toggle"),
    align: $("cmp-align"),
    meta: $("cmp-meta"),
  };

  const cstate = {
    requestSeq: 0,          // monotonic; older responses are dropped
    lastRequestId: null,
    result: null,           // last successfully rendered comparison payload
    // Context the rendered result was computed for.
    ctx: null,              // { sessionId, generation, revision, start, end, bFileName }
    expired: false,
    sampleB: null,          // frames_b loaded from a bundled demo, if any
  };

  const MIN_FRAMES = 8;
  const MAX_FRAMES = 64;

  function api(path, options = {}) {
    return window
      .fetch(path, { headers: { "Content-Type": "application/json" }, ...options })
      .then(async (response) => {
        let payload = null;
        try { payload = await response.json(); } catch { payload = null; }
        if (!response.ok) {
          const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
          const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
          err.status = response.status;
          throw err;
        }
        return payload;
      });
  }

  function showError(message) {
    if (!message) { els.error.hidden = true; els.error.textContent = ""; return; }
    els.error.hidden = false;
    els.error.textContent = message;
  }

  function currentSession() {
    const hook = window.CTC_WORKBENCH;
    return hook ? hook.getSession() : { sessionId: null, snapshot: null };
  }

  // ------------------------------------------------------------- range inputs

  function syncRangeLimits() {
    const { snapshot } = currentSession();
    const total = snapshot ? snapshot.frame_count : 0;
    const done = snapshot ? snapshot.next_frame : 0;
    els.start.max = String(Math.max(0, done));
    els.end.max = String(done);
    if (Number(els.start.value) > done) els.start.value = String(Math.max(0, done - MIN_FRAMES));
    if (Number(els.end.value) > done || Number(els.end.value) === 0) {
      els.end.value = String(Math.min(done, total));
    }
    clampRange();
  }

  function clampRange() {
    const { snapshot } = currentSession();
    const done = snapshot ? snapshot.next_frame : 0;
    let s = Number(els.start.value);
    let e = Number(els.end.value);
    s = Math.max(0, Math.min(s, done));
    e = Math.max(0, Math.min(e, done));
    if (e - s < MIN_FRAMES) {
      e = Math.min(done, s + MIN_FRAMES);
      if (e - s < MIN_FRAMES) s = Math.max(0, e - MIN_FRAMES);
    }
    if (e - s > MAX_FRAMES) e = s + MAX_FRAMES;
    els.start.value = String(s);
    els.end.value = String(e);
    const len = e - s;
    els.rangeOut.textContent = `${s} – ${e}（${len} 帧）`;
    updateRunButton();
    markExpiredIfChanged();
  }

  function hasBSource() {
    return !!cstate.sampleB || !!els.bFile.files[0];
  }

  function updateRunButton() {
    const { sessionId, snapshot } = currentSession();
    const done = snapshot ? snapshot.next_frame : 0;
    const len = Number(els.end.value) - Number(els.start.value);
    els.run.disabled =
      !sessionId || done < MIN_FRAMES || len < MIN_FRAMES || len > MAX_FRAMES || !hasBSource();
    if (done < MIN_FRAMES) {
      els.rangeHint.textContent = `已推进 ${done} 帧；A 段至少需要 ${MIN_FRAMES} 帧，请先推进。`;
    } else {
      els.rangeHint.textContent = "只能选择已经推进过的帧；换区间后旧结果会标记为过期。";
    }
  }

  // --------------------------------------------------------------- staleness

  function markExpiredIfChanged() {
    if (!cstate.result || cstate.expired) return;
    const { sessionId, generation, snapshot } = currentSession();
    const s = Number(els.start.value);
    const e = Number(els.end.value);
    const ctx = cstate.ctx;
    const stale =
      ctx.generation !== generation ||
      ctx.sessionId !== sessionId ||
      ctx.start !== s ||
      ctx.end !== e ||
      (snapshot && ctx.revision !== snapshot.revision);
    if (stale) setExpired(true);
  }

  function setExpired(on) {
    cstate.expired = on;
    renderBanner();
  }

  function renderBanner() {
    if (!cstate.result) { els.banner.hidden = true; return; }
    if (cstate.expired) {
      const ctx = cstate.ctx;
      els.banner.hidden = false;
      els.banner.innerHTML =
        `<strong>结果已过期</strong>：该结果基于会话修订 #${ctx.revision} 的 A 段帧范围 ` +
        `${ctx.start}–${ctx.end}（${ctx.end - ctx.start} 帧）。当前选择或进度已变化，` +
        "请重新对读以获取新结论；下表内容保留但不再作为当前证据。";
    } else {
      els.banner.hidden = true;
      els.banner.textContent = "";
    }
  }

  // --------------------------------------------------------------- rendering

  function statusLabel(status) {
    if (status === "ok") return { text: "可对应", cls: "good" };
    if (status === "conflict") return { text: "存在冲突", cls: "bad" };
    return { text: "证据不足", cls: "warn" };
  }

  function renderResult(payload) {
    cstate.result = payload;
    els.result.hidden = false;
    renderBanner();

    const st = statusLabel(payload.status);
    const sim = payload.similarity === null || payload.similarity === undefined
      ? "—"
      : payload.similarity.toFixed(4);
    const parts = payload.score_parts || {};
    const seg = payload.segment_a || {};
    const cards = [
      { k: "结论", v: st.text, cls: st.cls },
      { k: "相似度（0～1）", v: sim, cls: payload.status === "ok" ? "good" : "muted" },
      { k: "事件对应", v: `${parts.matched_events ?? "—"} / ${parts.total_events ?? "—"}`, cls: "muted" },
      { k: "A 段长度 / B 段长度", v: `${seg.length ?? payload.lengths.a} / ${payload.lengths.b}`, cls: "muted" },
    ];
    els.summary.innerHTML = "";
    cards.forEach((card) => {
      const div = document.createElement("div");
      div.className = "cmp-card";
      const k = document.createElement("span");
      k.className = "k";
      k.textContent = card.k;
      const v = document.createElement("span");
      v.className = `v ${card.cls}`;
      v.textContent = String(card.v);
      div.append(k, v);
      els.summary.appendChild(div);
    });

    if (payload.reason) {
      els.reason.hidden = false;
      els.reason.textContent = payload.reason;
    } else {
      els.reason.hidden = true;
      els.reason.textContent = "";
    }

    // Event pair table.
    els.pairsBody.innerHTML = "";
    const pairs = payload.pairs || [];
    els.pairsEmpty.hidden = pairs.length > 0;
    pairs.forEach((pair, idx) => {
      const tr = document.createElement("tr");
      if (!pair.tempo_ok) tr.className = "tempo-bad";
      const mk = (text) => {
        const td = document.createElement("td");
        td.textContent = text;
        return td;
      };
      tr.appendChild(mk(String(idx + 1)));
      tr.appendChild(mk(pair.token));
      tr.appendChild(
        mk(
          `${pair.a_start}-${pair.a_end}（绝对 ${seg.start_frame + pair.a_start}–` +
            `${seg.start_frame + pair.a_end}）`
        )
      );
      tr.appendChild(mk(`${pair.b_start}-${pair.b_end}`));
      tr.appendChild(mk(`${pair.a_frame_count} ↔ ${pair.b_frame_count}`));
      const speed = document.createElement("td");
      speed.className = "num";
      speed.textContent = pair.speed_ratio.toFixed(2) + "x";
      tr.appendChild(speed);
      const stateTd = document.createElement("td");
      const pill = document.createElement("span");
      if (pair.label_match && pair.tempo_ok) {
        pill.className = "pill ok";
        pill.textContent = "对应";
      } else if (!pair.label_match) {
        pill.className = "pill bad";
        pill.textContent = "发音冲突";
        tr.classList.add("pair-conflict");
      } else {
        pill.className = "pill bad";
        pill.textContent = "速度超限";
      }
      stateTd.appendChild(pill);
      tr.appendChild(stateTd);
      els.pairsBody.appendChild(tr);
    });

    // Conflicts.
    els.conflicts.innerHTML = "";
    const conflictList = payload.conflicts || [];
    els.conflictsEmpty.hidden = conflictList.length > 0;
    conflictList.forEach((c) => {
      const li = document.createElement("li");
      const tag = document.createElement("span");
      tag.className = "ctype";
      tag.textContent = c.type;
      li.append(tag, document.createTextNode(c.message));
      els.conflicts.appendChild(li);
    });

    renderAlignment(payload);

    const ctx = cstate.ctx;
    els.meta.textContent =
      `请求 ${payload.request_id || "—"} · 会话修订 #${payload.revision} · ` +
      `A 段原帧范围 ${ctx.start}–${ctx.end}（绝对） · B 段来源：${ctx.bFileName || "内置样例"} · ` +
      `对齐路径 ${payload.alignment.length} 个对应帧`;
  }

  function kindLabel(kind) {
    if (kind === "event") return "发音";
    if (kind === "blank") return "blank";
    return "模糊";
  }

  function renderAlignment(payload) {
    els.align.innerHTML = "";
    const seg = payload.segment_a || { start_frame: 0 };
    const path = payload.alignment || [];
    if (!path.length) {
      const empty = document.createElement("span");
      empty.className = "hint";
      empty.textContent = "无可用对齐路径。";
      els.align.appendChild(empty);
      return;
    }
    path.forEach((step) => {
      const row = document.createElement("div");
      const ka = payload.frame_kinds_a[step.a];
      const kb = payload.frame_kinds_b[step.b];
      row.className = `arow kind-${ka === kb ? ka : "ambiguous"}`;
      row.textContent =
        `A[${step.a}] 绝对帧 ${seg.start_frame + step.a}（${kindLabel(ka)}）  ↔  ` +
        `B[${step.b}]（${kindLabel(kb)}）`;
      els.align.appendChild(row);
    });
  }

  // ------------------------------------------------------------------- run

  function readBFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        let raw;
        try {
          raw = JSON.parse(reader.result);
        } catch (err) {
          reject(new Error(`B 段 JSON 解析失败：${err.message}`));
          return;
        }
        if (Array.isArray(raw)) { resolve(raw); return; }
        if (raw && Array.isArray(raw.frames)) { resolve(raw.frames); return; }
        if (raw && Array.isArray(raw.probs)) { resolve(raw.probs); return; }
        reject(new Error("B 段文件必须是帧数组，或含 frames / probs 字段的对象"));
      };
      reader.onerror = () => reject(new Error("B 段文件读取失败"));
      reader.readAsText(file);
    });
  }

  async function runCompare() {
    const { sessionId, generation, snapshot } = currentSession();
    if (!sessionId || !snapshot) {
      showError("请先创建会话并推进帧");
      return;
    }
    const start = Number(els.start.value);
    const end = Number(els.end.value);
    let framesB;
    let bFileName = cstate.sampleB ? "内置对读样例" : "";
    if (cstate.sampleB) {
      framesB = cstate.sampleB;
    } else {
      const file = els.bFile.files[0];
      if (!file) { showError("请选择 B 段概率文件，或点击内置对读样例"); return; }
      bFileName = file.name;
      try {
        framesB = await readBFile(file);
      } catch (err) {
        showError(err.message);
        return;
      }
    }

    const requestId = `cmp-${Date.now()}-${++cstate.requestSeq}`;
    cstate.lastRequestId = requestId;
    showError("");
    els.run.disabled = true;
    els.run.textContent = "对读计算中…";
    try {
      const payload = await api("/api/compare", {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          start_frame: start,
          end_frame: end,
          frames_b: framesB,
          request_id: requestId,
        }),
      });
      // Late response: a newer comparison has been started, or the session was
      // replaced.  Never overwrite the newer result.
      if (payload.request_id !== cstate.lastRequestId) return;
      const live = currentSession();
      if (payload.session_id !== live.sessionId) return;
      cstate.ctx = {
        sessionId,
        generation,
        revision: payload.revision,
        start,
        end,
        bFileName,
      };
      cstate.expired = false;
      renderResult(payload);
    } catch (err) {
      // Failure must preserve the previous result (if any); show the reason.
      if (cstate.result) {
        showError(`对读失败，已保留上次结果：${err.message}`);
      } else {
        showError(`对读失败：${err.message}`);
      }
    } finally {
      els.run.textContent = "开始对读";
      updateRunButton();
    }
  }

  // -------------------------------------------------------------- demo samples

  async function loadCompareSamples() {
    try {
      const payload = await api("/api/compare-samples");
      const items = payload.samples || [];
      els.samples.innerHTML = "";
      items.forEach((item) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample ghost";
        btn.textContent = item.title || item.name;
        btn.title = item.description || item.name;
        btn.addEventListener("click", () => loadPairSample(item.name, btn));
        els.samples.appendChild(btn);
      });
    } catch {
      /* samples are optional */
    }
  }

  async function loadPairSample(name, btn) {
    try {
      const data = await api(`/api/compare-samples/${encodeURIComponent(name)}`);
      cstate.sampleB = data.frames_b || data.frames;
      els.bFile.value = "";
      const range = data.suggested_range;
      const { snapshot } = currentSession();
      if (Array.isArray(range) && snapshot && range[1] <= snapshot.next_frame) {
        els.start.value = String(range[0]);
        els.end.value = String(range[1]);
      }
      clampRange();
      showError("");
      [...els.samples.children].forEach((b) => (b.style.borderColor = ""));
      btn.style.borderColor = "var(--accent)";
      updateRunButton();
    } catch (err) {
      showError(`载入对读样例失败：${err.message}`);
    }
  }

  // ------------------------------------------------------------------ wiring

  els.start.addEventListener("input", clampRange);
  els.end.addEventListener("input", clampRange);
  els.bFile.addEventListener("change", () => {
    cstate.sampleB = null;
    [...els.samples.children].forEach((b) => (b.style.borderColor = ""));
    showError("");
    updateRunButton();
  });
  els.run.addEventListener("click", runCompare);
  els.alignToggle.addEventListener("click", () => {
    const hidden = els.align.hidden;
    els.align.hidden = !hidden;
    els.alignToggle.textContent = hidden ? "收起" : "展开";
  });

  // The streaming workbench notifies us after every accepted snapshot.  A
  // generation change means the session was replaced: drop the old result and
  // selections.  A revision change on the same session merely expires it.
  let lastGeneration = currentSession().generation;
  window.CTC_WORKBENCH.onSnapshot = () => {
    const live = currentSession();
    if (live.generation !== lastGeneration) {
      lastGeneration = live.generation;
      cstate.result = null;
      cstate.ctx = null;
      cstate.expired = false;
      cstate.sampleB = null;
      els.result.hidden = true;
      els.bFile.value = "";
      els.start.value = "0";
      els.end.value = String(MIN_FRAMES);
      showError("");
    }
    syncRangeLimits();
    markExpiredIfChanged();
    updateRunButton();
  };

  syncRangeLimits();
  updateRunButton();
  loadCompareSamples();
})();
