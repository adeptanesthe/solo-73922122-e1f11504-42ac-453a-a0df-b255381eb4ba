/* Headless smoke test of the UI against a live server using jsdom.
 *
 * Usage: node tests/ui_smoke.mjs http://127.0.0.1:8123
 * Requires jsdom available from the working directory (see README).
 */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

// jsdom is an optional dev dependency; point JSDOM_DIR at a node_modules
// directory that contains it when it is not installed next to this file.
const require = createRequire(
  process.env.JSDOM_DIR ? path.join(process.env.JSDOM_DIR, "noop.js") : import.meta.url,
);
const { JSDOM, VirtualConsole } = require("jsdom");

const BASE = process.argv[2] || "http://127.0.0.1:8123";
const root = path.resolve(new URL("..", import.meta.url).pathname);

const html = fs.readFileSync(path.join(root, "app/static/index.html"), "utf8");
const script = fs.readFileSync(path.join(root, "app/static/app.js"), "utf8");

const virtualConsole = new VirtualConsole();
const errors = [];
virtualConsole.on("jsdomError", (e) => errors.push(String(e)));
virtualConsole.on("error", (e) => errors.push(String(e)));

const dom = new JSDOM(html, {
  url: BASE + "/",
  runScripts: "outside-only",
  virtualConsole,
  pretendToBeVisual: true,
});
const { window } = dom;
window.fetch = (input, init) => fetch(new URL(input, BASE), init);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, label, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return;
    await sleep(50);
  }
  throw new Error("timeout waiting for " + label);
}
const $ = (id) => window.document.getElementById(id);

window.eval(script);
await sleep(200);

const results = [];
function check(name, condition, detail = "") {
  results.push({ name, ok: !!condition, detail });
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
}

// 1. samples are listed
await until(() => $("sample-buttons").children.length > 0, "sample buttons");
check("样例按钮已渲染", $("sample-buttons").children.length >= 3,
  $("sample-buttons").textContent.trim());

// 2. load the repeat sample
const repeatBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sample_repeat"));
repeatBtn.click();
await until(() => !$("workspace").hidden && $("progress-text").textContent !== "0 / 0 帧", "sample loaded");
check("载入样例后显示工作区", !$("workspace").hidden, $("progress-text").textContent);
check("初始无候选字幕", $("confirmed-text").textContent === "", `“${$("confirmed-text").textContent}”`);

// 3. advance one chunk
window.document.getElementById("chunk-size").value = "5";
$("chunk-size").dispatchEvent(new window.Event("input"));
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("5 /"), "advance chunk");
check("分块推进更新进度", $("progress-text").textContent.startsWith("5 /"), $("progress-text").textContent);
check("候选表非空", $("beam-body").children.length > 0, `${$("beam-body").children.length} 束`);
const firstText = $("beam-body").children[0].querySelector(".beam-text").textContent;
check("最佳候选含字符", firstText.length > 0, firstText);

// 4. confirm the best candidate
$("confirm-best").click();
await until(() => $("confirmed-text").textContent.length > 0, "confirm");
const confirmed = $("confirmed-text").textContent;
check("确认后前缀显示", confirmed.length > 0, confirmed);
check("blank 分数显示", $("ps-blank").textContent !== "—", $("ps-blank").textContent);
check("非 blank 分数显示", $("ps-nonblank").textContent !== "—", $("ps-nonblank").textContent);
check("撤销按钮可用", !$("undo-btn").disabled);

// 5. advance after confirmation, then undo
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("10 /"), "advance after confirm");
$("undo-btn").click();
await until(() => $("confirmed-text").textContent === "", "undo");
check("撤销后前缀清空", $("confirmed-text").textContent === "");
check("撤销后进度保留", $("progress-text").textContent.startsWith("10 /"), $("progress-text").textContent);
check("撤销后候选非空", $("beam-body").children.length > 0);

// 6. failure keeps state: confirm text not in beam
$("prefix-input").value = "zzzz";
$("apply-prefix").click();
await until(() => !$("op-error").hidden, "error shown");
const beforeProgress = $("progress-text").textContent;
check("非法确认显示错误", !$("op-error").hidden, $("op-error").textContent);
check("错误后进度不变", $("progress-text").textContent === beforeProgress, $("progress-text").textContent);

// 7. historical frame inspection
$("peek-frame").value = "2";
$("peek-frame").dispatchEvent(new window.Event("input"));
await sleep(400);
check("历史帧提示只读", $("peek-note").textContent.includes("历史回放"), $("peek-note").textContent);
check("历史帧概率已渲染", $("frame-probs").children.length > 0);

// 8. continuous run to the end
$("peek-frame").value = $("peek-frame").max;
$("peek-frame").dispatchEvent(new window.Event("input"));
$("run-btn").click();
await until(() => $("progress-text").textContent.startsWith("15 / 15"), "run to end", 30000);
check("连续推进到结尾", $("progress-text").textContent.startsWith("15 / 15"), $("progress-text").textContent);
await until(() => $("stop-btn").hidden, "run loop finished");
check("运行结束停止按钮隐藏", $("stop-btn").hidden);

// 9. zero-probability sample shows -∞ without NaN
const zeroBtn = [...$("sample-buttons").children].find((b) => b.textContent.includes("sparse_zero"));
zeroBtn.click();
await until(() => $("progress-text").textContent.startsWith("0 / 7"), "zero sample loaded");
$("step-btn").click();
await until(() => $("progress-text").textContent.startsWith("7 / 7") || $("progress-text").textContent.startsWith("5 / 7"), "zero sample advance");
await sleep(300);
const bodyText = $("beam-body").textContent;
check("零概率样例无 NaN", !bodyText.includes("NaN"), bodyText.slice(0, 120));
check("零概率样例显示 -∞", bodyText.includes("-∞") || $("frame-probs").textContent.includes("0.000"));

// 10. no uncaught UI errors so far
check("此前无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

// ===========================================================================
// 11. two-segment duplicate-check panel
// ===========================================================================
// Use the duplicate pair demo: its frames_a doubles as a fresh session.
const pairSamples = await (await fetch(new URL("/api/compare-samples", BASE))).json();
check("对读样例接口可用", pairSamples.samples.length >= 2, pairSamples.samples.map((s) => s.name).join(","));
const dupSample = await (
  await fetch(new URL("/api/compare-samples/pair_duplicate.json", BASE))
).json();
// Create a matching-vocabulary session through the API and adopt it in the
// page via the explicit test hook, then drive the real UI to advance frames.
const pairSession = await (
  await fetch(new URL("/api/sessions", BASE), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vocab: dupSample.vocab, beam_width: 4, frames: dupSample.frames_a }),
  })
).json();
window.CTC_WORKBENCH.__testAdoptSession(pairSession, "pair_duplicate");
await until(() => !$("workspace").hidden, "pair session adopted");
$("run-btn").click();
await until(
  () => $("progress-text").textContent.startsWith(`${dupSample.frames_a.length} / ${dupSample.frames_a.length}`),
  "advance pair session",
  30000,
);
await until(() => $("stop-btn").hidden, "run loop finished 2");

check("对读面板存在", !!$("compare-panel"));
await until(() => $("cmp-sample-buttons").children.length >= 2, "compare sample buttons");
check("对读样例按钮已渲染", $("cmp-sample-buttons").children.length >= 2);
check("未选 B 段时对读按钮禁用", $("cmp-run").disabled);

const dupBtn = [...$("cmp-sample-buttons").children].find((b) => b.textContent.includes("重复采集"));
dupBtn.click();
await sleep(200);
check("载入对读样例后按钮可用", !$("cmp-run").disabled, `range=${$("cmp-range-out").textContent}`);
$("cmp-run").click();
await until(() => !$("cmp-result").hidden && $("cmp-summary").children.length > 0, "compare result");
check("对读结果区显示", !$("cmp-result").hidden);
const verdict = $("cmp-summary").children[0].textContent;
check("重复样例结论可对应", verdict.includes("可对应"), verdict);
const simText = $("cmp-summary").children[1].textContent;
check("相似度为 0~1 数值", /\d\.\d{4}/.test(simText), simText);
check("重复样例无冲突", $("cmp-conflicts").children.length === 0, $("cmp-conflicts").textContent);
check("事件对应表非空", $("cmp-pairs-body").children.length >= 1);
check("对齐路径默认收起", $("cmp-align").hidden);
$("cmp-align-toggle").click();
check("展开对齐路径", !$("cmp-align").hidden && $("cmp-align").children.length > 0);
$("cmp-align-toggle").click();
check("再次收起对齐路径", $("cmp-align").hidden);
check("未显示过期横幅", $("cmp-banner").hidden);

// Advancing is impossible at the end; switch the A range instead to force expiry.
const oldEnd = Number($("cmp-end").value);
$("cmp-start").value = String(Math.max(0, oldEnd - 10));
$("cmp-start").dispatchEvent(new window.Event("input"));
await sleep(100);
check("更换 A 段区间后旧结果标过期", !$("cmp-banner").hidden && $("cmp-banner").textContent.includes("结果已过期"),
  $("cmp-banner").textContent.slice(0, 60));
check("过期横幅含原帧范围", /帧范围/.test($("cmp-banner").textContent));

// Conflict demo must surface conflicts rather than a similarity verdict.
const conBtn = [...$("cmp-sample-buttons").children].find((b) => b.textContent.includes("冲突"));
conBtn.click();
await sleep(150);
$("cmp-run").click();
await until(() => $("cmp-conflicts").children.length > 0, "conflict result");
check("冲突样例列出冲突事件", $("cmp-conflicts").children.length > 0, $("cmp-conflicts").textContent.slice(0, 80));
check("冲突样例结论为冲突", $("cmp-summary").children[0].textContent.includes("冲突"));
check("新结果不再显示过期", $("cmp-banner").hidden);

// A failed comparison (all-zero B frame via an uploaded file) keeps the old result.
const zeroFrames = [...dupSample.frames_b.slice(0, -1), [0, 0, 0, 0]];
const zeroFile = new window.File([JSON.stringify({ frames: zeroFrames })], "zero.json", {
  type: "application/json",
});
const cmpFile = $("cmp-b-file");
Object.defineProperty(cmpFile, "files", { value: [zeroFile], configurable: true });
cmpFile.dispatchEvent(new window.Event("change"));
await sleep(100);
$("cmp-run").click();
await until(() => !$("cmp-error").hidden && $("cmp-error").textContent.includes("全零帧"), "zero B error");
check("全零帧报错并保留上次结果", $("cmp-error").textContent.includes("保留上次结果"), $("cmp-error").textContent);
check("报错后旧结果仍在屏幕上", !$("cmp-result").hidden && $("cmp-conflicts").children.length > 0);

// 12. no uncaught UI errors
check("无未捕获脚本错误", errors.length === 0, errors.join(" | ").slice(0, 200));

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
