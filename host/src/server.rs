//! OpenAI-compatible HTTP server (std only): GET /v1/models, POST /v1/chat/completions.
//!
//! Threads: the main thread owns the GPU engine and runs the scheduler; an acceptor thread starts one thread per
//! connection. A connection thread parses the request, renders and tokenises the prompt, hands a `Job` to the
//! scheduler and then turns the token batches it receives into text and server-sent events - so a slow client
//! never stalls the engine.
//!
//! Scheduler: every engine step carries rows from all active conversations. Conversations that are generating
//! contribute their next token (plus drafts worth verifying); conversations still reading their prompt share the
//! remaining row budget.
use crate::{chat::Chat, engine::{sampled_drafts, Drafts, Engine, SpecStats, DRAFT_BATCH, TMAX}};
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    io::{BufRead, BufReader, Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc, Arc, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_CONNECTIONS: usize = 256; // open connections (each has a thread); more are refused with 503
const MAX_BODY: usize = 32 << 20; // refuse larger request bodies instead of allocating whatever the header says
const EOS: u32 = 200_006;
const T_THINK: u32 = 200_008;
const T_END: u32 = 200_010;
const T_MODEL: u32 = 200_001; // <|message_model|>: starts a message of the model; a tool call puts the tool's name right after it
const T_TOOL_JSON: u32 = 200_049; // <|content_invoke_tool_json|>: the message body is {"name": ..., "args": {...}}
const FIRST_SPECIAL: u32 = 199_998;

/// Makes tool-call ids unique. The chat template pairs a tool result with its call by id, so two calls sharing an id
/// would garble the next prompt.
static CALL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Server settings (see `kuroshio --help`).
#[derive(Clone)]
pub struct Options {
    /// Speculative decoding: verify up to `k_max` drafts per step...
    pub k_max: usize,
    /// ...and only those the drafter gives probability >= `tau`. None: decide per step by expected gain against the
    /// cost of a row (see `Speculation`).
    pub tau: Option<f32>,
    /// Prompt rows per step while some conversation is generating: bounds how long a step (and so everyone's next
    /// token) takes.
    pub prefill_share: usize,
    /// Keep each slot's state so that a request sharing a prefix with the slot's last one skips that prefix.
    pub prefix_cache: bool,
    /// Supply the fixed beginning of a tool call instead of letting the model write it.
    pub tool_fix: bool,
    /// Origin that browsers are told may call this server (`*` for any); None: no cross-origin access.
    pub cors: Option<String>,
    /// Give back the context memory of a slot that has been idle this many seconds (0: never). Its prefix cache goes too.
    pub idle_release: u64,
    /// Verify up to this many tokens copied from the conversation itself when it starts repeating an earlier passage
    /// (0: off). See `Copy`.
    pub copy_max: usize,
}

/// What the scheduler sends back to a connection thread.
enum Event {
    Tokens(Vec<u32>),
    Done { finish: &'static str, completion_tokens: usize, cached: usize, summary: String },
    /// The request cannot be served (e.g. the GPU has no memory left for its context).
    Failed(String),
}

/// A request waiting for or occupying a slot.
struct Job {
    prompt: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    seed: u32,
    tx: mpsc::Sender<Event>,
    t0: Instant,
    /// Set by the connection thread when it stops waiting (client gone, error): the scheduler then drops the job.
    cancel: Arc<AtomicBool>,
}

/// Sets the flag when the connection thread leaves `completion`, however it leaves.
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// xorshift64* - enough for acceptance tests and ids; avoids a dependency.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ------------------------------------------------------------------------------------------- scheduler
enum Phase {
    /// `done` prompt tokens have been processed.
    Prefill { done: usize },
    /// `anchor` is the last emitted token, not yet run through the model.
    Decode { anchor: u32 },
}

struct Active {
    job: Job,
    phase: Phase,
    pos: usize, // next position to process
    n_out: usize,
    rng: Rng,
    stats: SpecStats,
    ttft: f64,
    cached: usize, // prompt tokens reused from this slot's previous conversation
    /// Tokens of the message header being generated (between <|message_model|> and the content-type token).
    header: Option<Vec<u32>>,
    /// Tokens that must come next (the fixed beginning of a tool call); they are fed to the model, not sampled.
    forced: Vec<u32>,
    copy: Copy,
    /// Tokens accepted from copied drafts.
    copied: usize,
    /// Drafts the drafter has been getting accepted per step lately: what copying has to beat.
    draft_yield: f32,
}

/// Follows the tokens a conversation is about to emit. When the model opens a tool call
/// (`<|message_model|>NAME<|content_invoke_tool_json|>`), the body must begin `{"name":"NAME","args":` - the chat
/// template allows nothing else. Inkling 2 Mini gets exactly that stretch wrong surprisingly often (after the name it
/// puts ~80% on skipping the `,"`; vLLM shows the same malformed calls), so the server supplies it: the stretch is
/// queued in `forced` and everything after the tool-call token in `out` is dropped. Returns the new length of `out`.
fn watch_tool_call(a: &mut Active, out: &[u32], chat: &Chat) -> usize {
    for (j, &t) in out.iter().enumerate() {
        if t == T_MODEL {
            a.header = Some(Vec::new());
        } else if t == T_TOOL_JSON {
            if let Some(h) = a.header.take().filter(|h| !h.is_empty()) {
                let name = chat.decode(&h).unwrap_or_default();
                let text = format!("{{\"name\":{},\"args\":", serde_json::to_string(&name).unwrap_or_default());
                // The name is the model's text, not markup: it must not turn into control tokens on the way back in.
                if let (false, Ok(toks)) = (name.is_empty(), chat.encode_text(&text)) {
                    a.forced = toks;
                    return j + 1;
                }
            }
        } else if t >= FIRST_SPECIAL {
            a.header = None;
        } else if let Some(h) = a.header.as_mut() {
            h.push(t);
            if h.len() > 64 {
                a.header = None; // not a tool name
            }
        }
    }
    out.len()
}

/// Drafts copied from the conversation itself. Models repeat their context all the time - quoting a file they were
/// shown, rewriting code with one line changed, echoing a tool's output. When the last tokens match an earlier passage,
/// the tokens that followed it there are a free draft: no drafter pass, any length, and very often right.
///
/// `seq` is everything in the conversation so far (prompt, then generated tokens); `idx` maps each run of three tokens
/// to where it last ended. `src` is the earlier position currently being followed and `run` how many tokens have
/// matched up to now.
///
/// How often copied tokens are accepted differs a lot between conversations (one repeats a file verbatim, another
/// only quotes a phrase and moves on), so each request keeps its own count, by length of the match.
struct Copy {
    seq: Vec<u32>,
    idx: HashMap<(u32, u32, u32), u32>,
    src: Option<usize>,
    run: usize,
    acc: [f32; 4],
    ver: [f32; 4],
}

impl Copy {
    fn new(prompt: &[u32]) -> Self {
        let mut c = Self { seq: Vec::with_capacity(prompt.len() + 1024), idx: HashMap::with_capacity(prompt.len()), src: None, run: 0, acc: [0.0; 4], ver: [0.0; 4] };
        for &t in prompt {
            c.push(t);
        }
        c
    }

    fn push(&mut self, t: u32) {
        let i = self.seq.len();
        self.seq.push(t);
        // Still agreeing with the passage being followed?
        match self.src {
            Some(s) if s + 1 < i && self.seq[s + 1] == t => (self.src, self.run) = (Some(s + 1), self.run + 1),
            _ => (self.src, self.run) = (None, 0),
        }
        if i >= 2 {
            let earlier = self.idx.insert((self.seq[i - 2], self.seq[i - 1], t), i as u32);
            if let (None, Some(p)) = (self.src, earlier) {
                let p = p as usize;
                let n = (0..64.min(p + 1)).take_while(|&n| self.seq[p - n] == self.seq[i - n]).count();
                (self.src, self.run) = (Some(p), n);
            }
        }
    }

    /// Matches of 4-5, 6-9, 10-19 and 20+ tokens. Shorter ones are not used: three tokens in a row recur by chance.
    fn bucket(run: usize) -> usize {
        match run {
            0..=5 => 0,
            6..=9 => 1,
            10..=19 => 2,
            _ => 3,
        }
    }

    /// Probability that a copied token is accepted (given the ones before it were), for a match of `run` tokens:
    /// what this request has shown so far, starting from a guess worth 8 observations.
    fn accept(&self, run: usize) -> f32 {
        const PRIOR: [f32; 4] = [0.55, 0.7, 0.85, 0.92];
        let b = Self::bucket(run);
        ((self.acc[b] + 8.0 * PRIOR[b]) / (self.ver[b] + 8.0)).clamp(0.0, 0.995)
    }

    fn observe(&mut self, run: usize, accepted: bool) {
        let b = Self::bucket(run);
        self.ver[b] = self.ver[b] * 0.98 + 1.0;
        self.acc[b] = self.acc[b] * 0.98 + accepted as u8 as f32;
    }

    /// Is copying the better draft right now? Compares the tokens it should yield (`max` drafts, each accepted with
    /// the probability seen so far) with what the drafter has been yielding per step in this request.
    fn worth(&self, max: usize, drafter_yield: f32) -> bool {
        let p = self.accept(self.run);
        let expected: f32 = (1..=max as i32).map(|j| p.powi(j)).sum();
        self.run >= 4 && self.src.is_some() && expected > drafter_yield
    }

    /// Called when a match was not used: what was learnt about matches of this length fades, so that copying is
    /// tried again after a while (the text may have changed character).
    fn forget(&mut self) {
        let b = Self::bucket(self.run);
        self.ver[b] *= 0.9;
        self.acc[b] *= 0.9;
    }

    /// What followed the matched passage: the proposed continuation, at most `max` tokens. When the passage is the
    /// text just written (a repeating pattern: table rules, a run of zeros), the continuation runs into the drafts
    /// themselves, which is what a repeat of period P means.
    fn drafts(&self, max: usize) -> Vec<u32> {
        let Some(s) = self.src else { return Vec::new() };
        let period = self.seq.len() - (s + 1);
        let mut out = Vec::with_capacity(max);
        for j in 0..max {
            out.push(if j < period { self.seq[s + 1 + j] } else { out[j - period] });
        }
        out
    }
}

/// A conversation slot. It keeps what its last request left in the engine's caches, so that a new request sharing a
/// prefix with it does not have to process that prefix again.
struct Slot {
    /// Tokens at positions 0..hist.len() of this slot's caches.
    hist: Vec<u32>,
    /// A snapshot of the slot's ring buffers exists for this position (the end of the last prompt, or one token
    /// before it for long prompts so that the same prompt sent again can use it too).
    ckpt: Option<usize>,
    /// Positions 0..hi of the slot's ring buffers have been written at some point, rejected drafts included; `ckpt_hi`
    /// is the same for the snapshot.
    hi: usize,
    ckpt_hi: usize,
    /// Tokens of context this slot has GPU memory for. It is attached as the conversation grows.
    cap: usize,
    /// The last attempt to attach more failed (the GPU is full): the conversation ends when it reaches `cap`.
    full: bool,
    last_used: Instant,
    active: Option<Active>,
}

/// Position at which a prompt's snapshot is taken: its end, or for long prompts one token earlier, so that the very
/// same prompt sent again (a retry, a regeneration) still has a token left to process and can use the snapshot.
fn snapshot_at(prompt_len: usize) -> usize {
    if prompt_len >= 512 { prompt_len - 1 } else { prompt_len }
}

/// Where could a request with this prompt pick up in this slot? Returns (position, needs_restore).
/// - "live": the prompt continues (almost) exactly where the slot's caches end;
/// - "snapshot": the prompt shares at least the previous prompt, whose end state was saved.
/// At least one prompt token must remain to be processed, so the position is at most prompt.len() - 1.
fn resume_point(slot: &Slot, prompt: &[u32]) -> (usize, bool) {
    let lcp = slot.hist.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let mut best = (0usize, false);
    let live = lcp.min(prompt.len() - 1);
    // The convolution history is a ring, and a long run of rows keeps only its last `max_spec_run` positions in it.
    // Resuming at `live` needs the 3 positions before it, so nothing may have been written `max_spec_run - 3` or
    // more positions past them - neither accepted tokens nor rejected drafts.
    if live > 0 && slot.hi <= live + Engine::max_spec_run() - 3 {
        best = (live, false);
    }
    if let Some(c) = slot.ckpt {
        if lcp >= c && c < prompt.len() && c > best.0 {
            best = (c, true);
        }
    }
    best
}

/// Fixed-threshold speculation (`--tau X`): how many drafts to try, by number of conversations generating at once.
/// Measured aggregate tokens/s at 2 / 4 / 8 streams (temperature 0.6, tau 0.5): no speculation 256 / 394 / 576;
/// up to 3-4 drafts 362-380 / 465-470 / 622-625; up to 7 drafts 382 / 452 / 588.
fn fixed_k(cfg: &Options, decoders: usize, multi: Option<(usize, f32)>) -> (usize, f32) {
    let tau = cfg.tau.unwrap_or(0.5);
    if let (Some(m), true) = (multi, decoders > 1) {
        return m;
    }
    match decoders {
        0..=2 => (cfg.k_max, tau),
        3 | 4 => (cfg.k_max.min(4), tau),
        _ => (cfg.k_max.min(3), tau),
    }
}

/// What the scheduler has learned about speculation while serving, used to decide how many drafts to verify.
///
/// A verified draft is a row of the step: it costs time whether or not it is accepted, and it only yields a token if
/// every draft before it was accepted too. So draft j of a conversation is worth `P(drafts 1..j all accepted)` tokens
/// and costs one more row. Rows are added, best first across all conversations, while that raises tokens per
/// millisecond - which automatically means fewer drafts when many conversations share the step.
struct Speculation {
    /// Accepted / verified counts by drafter confidence (10 buckets), for most-likely-token and for sampled drafts.
    acc: [[f32; 10]; 2],
    ver: [[f32; 10]; 2],
    /// Measured step time relative to the table in `rows_ms`, and the time one drafter pass takes.
    scale: f32,
    draft_ms: f32,
}

impl Speculation {
    fn new() -> Self {
        Self { acc: [[0.0; 10]; 2], ver: [[0.0; 10]; 2], scale: 1.0, draft_ms: 0.8 }
    }

    /// Step time in ms by number of rows, output head included, as measured on an RTX 5090 at short context
    /// (`kuroshio steptime` with INK_STEP_T; linear between the points). Not smooth: the matrix kernels work on
    /// tiles of rows, and the 9th, 17th, 33rd and 65th row each start a new one.
    fn rows_ms(rows: usize) -> f32 {
        const PTS: [(f32, f32); 46] = [
            (1.0, 6.12), (2.0, 6.97), (3.0, 7.71), (4.0, 8.40), (5.0, 9.01), (6.0, 9.61), (7.0, 10.19), (8.0, 10.69),
            (9.0, 11.57), (10.0, 12.13), (11.0, 12.59), (12.0, 13.00), (13.0, 13.37), (14.0, 13.90), (15.0, 14.37), (16.0, 14.97),
            (17.0, 16.27), (18.0, 16.69), (19.0, 17.17), (20.0, 17.62), (21.0, 17.92), (22.0, 18.46), (23.0, 18.92), (24.0, 19.30),
            (25.0, 19.73), (26.0, 20.19), (27.0, 20.61), (28.0, 21.17), (29.0, 21.56), (30.0, 22.27), (31.0, 22.73), (32.0, 23.03),
            (33.0, 26.07), (34.0, 26.42), (40.0, 28.62), (41.0, 29.16), (48.0, 31.96), (49.0, 32.67), (56.0, 35.56), (64.0, 39.98),
            (65.0, 41.82), (72.0, 44.81), (80.0, 48.69), (96.0, 56.49), (112.0, 65.19), (128.0, 72.32),
        ];
        let r = (rows.max(1) as f32).min(128.0);
        let i = PTS.iter().position(|p| p.0 >= r).unwrap_or(PTS.len() - 1).max(1);
        let ((x0, y0), (x1, y1)) = (PTS[i - 1], PTS[i]);
        y0 + (y1 - y0) * (r - x0) / (x1 - x0)
    }

    /// Probability that a draft is accepted, given the ones before it were, from the drafter's top probability at that
    /// position (never from the drafted token's own probability: for a sampled draft that would bias the output).
    /// Starts from the top probability itself and follows what is observed.
    fn accept(&self, conf: f32, sampled: bool) -> f32 {
        let (m, b) = (sampled as usize, ((conf * 10.0) as usize).min(9));
        let prior = if sampled { conf.max(0.5) } else { conf };
        ((self.acc[m][b] + 8.0 * prior) / (self.ver[m][b] + 8.0)).clamp(0.0, 1.0)
    }

    fn observe(&mut self, conf: f32, sampled: bool, accepted: bool) {
        let (m, b) = (sampled as usize, ((conf * 10.0) as usize).min(9));
        self.ver[m][b] = self.ver[m][b] * 0.998 + 1.0; // recent steps count more: text changes
        self.acc[m][b] = self.acc[m][b] * 0.998 + accepted as u8 as f32;
    }

    fn observe_step(&mut self, rows: usize, ms: f32) {
        self.scale = (self.scale * 0.98 + 0.02 * ms / Self::rows_ms(rows)).clamp(0.5, 4.0);
    }

    /// How many drafts to verify for each conversation. `drafts`: per conversation, its slot and for each of its
    /// drafts in order the probability that it is accepted given the ones before it were; `base_rows`: rows the step
    /// has anyway; `streams`: conversations that emit a token whatever happens; `prompt_rows`: rows of the step that
    /// read prompts (a longer step delays them too); `draft_pass`: a drafter pass was run.
    ///
    /// A draft row is worth P(it and all drafts before it are accepted) tokens and costs the step the time of one
    /// more row; the best n rows are verified, for the n that gives the step the most tokens per millisecond. This maximises the
    /// server's total output. It is not fair between conversations: one that is copying a long passage takes many
    /// rows, and the others' tokens arrive more slowly while it does. (Weighing gains in proportion to each
    /// conversation's own speed was tried: 4 streams lost up to 11% in total.)
    fn choose(&self, drafts: &[(usize, Vec<f32>)], base_rows: usize, streams: usize, prompt_rows: usize, nslots: usize, draft_pass: bool) -> Vec<usize> {
        let mut k = vec![0usize; nslots];
        let mut cands: Vec<(f32, usize)> = Vec::new();
        for (slot, probs) in drafts {
            let mut all = 1.0f32;
            for p in probs {
                all *= p;
                cands.push((all, *slot));
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)); // stable: a conversation's drafts stay in order
        // A prompt token counts for the time it would take alone, relative to an output token: 57 ms / 128 against 6.1 ms.
        const PROMPT_TOKEN: f32 = 0.073;
        // Take the best n rows, for the n that gives the most tokens per millisecond. (Not "add rows while they pay":
        // step time jumps at tile boundaries, and the rows after a jump can be worth it together when the first is not.)
        let base = base_rows.max(1);
        let fixed = if draft_pass { self.draft_ms } else { 0.0 };
        let mut tokens = streams.max(1) as f32 + PROMPT_TOKEN * prompt_rows as f32;
        let (mut best, mut best_n) = (tokens / (self.scale * Self::rows_ms(base) + fixed), 0);
        for (n, (all, _)) in cands.iter().enumerate().take(TMAX.saturating_sub(base)) {
            tokens += all;
            let rate = tokens / (self.scale * Self::rows_ms(base + n + 1) + fixed);
            if rate > best {
                (best, best_n) = (rate, n + 1);
            }
        }
        let rows = base + best_n;
        let taken: Vec<usize> = cands[..best_n].iter().map(|c| c.1).collect();
        // Every distinct number of rows is a recorded step graph (GPU memory, and milliseconds to record again once it
        // has been dropped), so big steps come in a few sizes only. The rows given up are the least promising ones.
        let keep = taken.len() - (rows - Self::step_size(rows).max(base_rows.max(1))).min(taken.len());
        for &slot in &taken[..keep] {
            k[slot] += 1;
        }
        k
    }

    /// The step sizes in use: any up to 16 rows, then multiples of 4 up to 64, then multiples of 8. Rounding down
    /// also lands big steps on full tiles of rows (see `rows_ms`).
    fn step_size(rows: usize) -> usize {
        match rows {
            0..=16 => rows,
            17..=64 => rows / 4 * 4,
            _ => rows / 8 * 8,
        }
    }
}

/// One conversation's part of a step.
struct Plan {
    r0: usize,
    n: usize,
    /// Tokens fed after the anchor: drafts to verify, or (`forcing`) text the server supplies.
    fed: Vec<u32>,
    forcing: bool,
    confs: [f32; 7],
    /// The drafter's top probability at each draft position (what the decision to verify was based on).
    tops: [f32; 7],
    /// The drafts were sampled from the drafter's distribution (accepted with min(1, p/q)).
    sampled: bool,
    /// The drafts were copied from the conversation; the length of the match that proposed them.
    copy_run: Option<usize>,
}

fn finish(a: Active, why: &'static str) {
    let dt = a.job.t0.elapsed().as_secs_f64() - a.ttft;
    let steps = a.stats.steps.max(1) as f64;
    let summary = format!(
        "{} prompt tokens ({} cached), ttft {:.0} ms; {} tokens at {:.1} tok/s; {:.2} tokens/step ({:.2} drafts verified, {:.2} accepted, {} tokens copied); finish={why}",
        a.job.prompt.len(), a.cached, a.ttft * 1e3, a.n_out, (a.n_out.max(2) - 1) as f64 / dt.max(1e-9), a.n_out as f64 / steps,
        a.stats.verified as f64 / steps, a.stats.accepted as f64 / steps, a.copied
    );
    // Nobody is listening to a request whose client went away: log it here.
    if let Err(mpsc::SendError(Event::Done { summary, .. })) = a.job.tx.send(Event::Done { finish: why, completion_tokens: a.n_out, cached: a.cached, summary }) {
        eprintln!("request: {summary}");
    }
}

/// Runs forever on the thread that owns the engine.
fn scheduler(eng: &mut Engine, rx: mpsc::Receiver<Job>, cfg: Options, max_ctx: usize, chat: &Chat) {
    let nslots = eng.slots();
    let mut spec = Speculation::new();
    let mut slots: Vec<Slot> = (0..nslots).map(|_| Slot { hist: Vec::new(), ckpt: None, hi: 0, ckpt_hi: 0, cap: 0, full: false, last_used: Instant::now(), active: None }).collect();
    let (prefix_cache, force_tools) = (cfg.prefix_cache, cfg.tool_fix);
    // Experiment knob: INK_SPEC_MULTI=k,tau overrides the load-based speculation policy when several streams generate.
    let multi: Option<(usize, f32)> = std::env::var("INK_SPEC_MULTI").ok().and_then(|v| {
        let (k, t) = v.split_once(',')?;
        Some((k.parse().ok()?, t.parse().ok()?))
    });
    // Conversations drafted per drafter pass (1 = one at a time).
    let draft_batch: usize = std::env::var("INK_DRAFT_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DRAFT_BATCH).clamp(1, DRAFT_BATCH);
    let mut queue: VecDeque<Job> = VecDeque::new();
    let prefill_share = cfg.prefill_share.clamp(1, TMAX);
    loop {
        // ---- new requests
        if slots.iter().all(|s| s.active.is_none()) && queue.is_empty() {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(j) => queue.push_back(j),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
        // Slots nobody has used for a while hand their context memory back to the GPU.
        if cfg.idle_release > 0 {
            for (i, sl) in slots.iter_mut().enumerate() {
                if sl.active.is_none() && sl.cap > 0 && sl.last_used.elapsed().as_secs() >= cfg.idle_release {
                    eng.trim(i, 0);
                    eprintln!("slot {i}: idle, released the memory of {} tokens of context", sl.cap - eng.ctx_mapped(i));
                    (sl.cap, sl.full, sl.ckpt, sl.hi) = (eng.ctx_mapped(i), false, None, 0);
                    sl.hist.clear();
                }
            }
        }
        while let Ok(j) = rx.try_recv() {
            queue.push_back(j);
        }
        // Requests whose client stopped waiting: forget them. A slot keeps what it has computed so far.
        queue.retain(|j| !j.cancel.load(Ordering::Relaxed));
        for sl in slots.iter_mut() {
            if sl.active.as_ref().map_or(false, |a| a.job.cancel.load(Ordering::Relaxed)) {
                finish(sl.active.take().unwrap(), "cancelled");
                sl.last_used = Instant::now();
            }
        }
        // Admit waiting requests. Each goes to the idle slot whose caches share the longest usable prefix with its
        // prompt; without any match, to the slot that has been idle longest.
        while !queue.is_empty() && slots.iter().any(|s| s.active.is_none()) {
            let job = queue.pop_front().unwrap();
            let mut pick: Option<(usize, usize, bool)> = None; // (slot, resume position, restore snapshot)
            for (i, sl) in slots.iter().enumerate().filter(|(_, sl)| sl.active.is_none()) {
                let (r, restore) = if prefix_cache { resume_point(sl, &job.prompt) } else { (0, false) };
                let better = match pick {
                    None => true,
                    Some((pi, pr, _)) => r > pr || (r == pr && r == 0 && sl.last_used < slots[pi].last_used),
                };
                if better {
                    pick = Some((i, r, restore));
                }
            }
            let (i, mut r, restore) = pick.unwrap();
            if std::env::var("INK_DEBUG_CACHE").is_ok() {
                for (j, sl) in slots.iter().enumerate() {
                    let lcp = sl.hist.iter().zip(&job.prompt).take_while(|(a, b)| a == b).count();
                    eprintln!("cache: slot {j} busy={} hist={} ckpt={:?} lcp={lcp} prompt={} -> pick slot {i} resume {r} restore={restore}; prompt[lcp..]={:?} hist[lcp..]={:?}",
                        sl.active.is_some(), sl.hist.len(), sl.ckpt, job.prompt.len(), &job.prompt[lcp..(lcp + 6).min(job.prompt.len())], &sl.hist[lcp..(lcp + 6).min(sl.hist.len())]);
                }
            }
            if restore && !eng.restore(i) {
                r = 0;
            }
            let sl = &mut slots[i];
            if restore {
                sl.hi = sl.ckpt_hi;
            } else if r == 0 {
                sl.hi = 0; // a fresh start rewrites every position before reading it
            }
            sl.hist.truncate(r);
            if sl.ckpt.map_or(false, |c| c > r) {
                sl.ckpt = None; // the saved state lies beyond what is kept
            }
            // A much shorter conversation moving into the slot: give the excess memory back.
            let keep = r.max(job.prompt.len()) + TMAX + 16;
            if sl.cap > keep + 8192 {
                eng.trim(i, keep);
                sl.cap = eng.ctx_mapped(i);
            }
            sl.full = false;
            let rng = Rng((job.seed as u64) << 20 | 1);
            let copy = if cfg.copy_max > 0 && cfg.k_max > 0 && eng.has_draft() { Copy::new(&job.prompt) } else { Copy::new(&[]) };
            sl.active = Some(Active { job, phase: Phase::Prefill { done: r }, pos: r, n_out: 0, rng, stats: SpecStats::default(), ttft: 0.0, cached: r, header: None, forced: Vec::new(), copy, copied: 0, draft_yield: 1.0 });
        }

        // ---- memory: every active conversation needs room for this step's positions. Attach more as it grows; if the
        // GPU has none, the conversation can go on only as far as what it already has.
        for i in 0..nslots {
            let Some(a) = slots[i].active.as_ref() else { continue };
            // A prompt asks for all it needs at once, so one that cannot fit is refused before any work is done.
            let ahead = match a.phase {
                Phase::Prefill { done } => a.job.prompt.len() - done,
                Phase::Decode { .. } => 0,
            };
            let want = (a.pos + ahead + TMAX + 16).min(max_ctx);
            if want > slots[i].cap {
                let mut got = eng.ensure(i, want);
                // The GPU is full: idle slots give up their caches, longest idle first, until this one fits.
                while got.is_none() {
                    let Some(j) = (0..nslots).filter(|&j| slots[j].active.is_none() && slots[j].cap > 0).min_by_key(|&j| slots[j].last_used) else { break };
                    eng.trim(j, 0);
                    eprintln!("slot {j}: released the memory of {} tokens of context for slot {i}", slots[j].cap - eng.ctx_mapped(j));
                    (slots[j].cap, slots[j].full, slots[j].ckpt, slots[j].hi) = (eng.ctx_mapped(j), false, None, 0);
                    slots[j].hist.clear();
                    got = eng.ensure(i, want);
                }
                match got {
                    Some(c) => (slots[i].cap, slots[i].full) = (c, false),
                    None => {
                        if !slots[i].full {
                            eprintln!("slot {i}: the GPU has no memory to extend this conversation beyond {} tokens", slots[i].cap);
                        }
                        slots[i].full = true;
                    }
                }
            }
            // Out of room: a prompt that cannot fit is refused, a generation ends here.
            let sl = &mut slots[i];
            let a = sl.active.as_ref().unwrap();
            let stuck = match a.phase {
                Phase::Prefill { done } => sl.full && a.pos + (a.job.prompt.len() - done) + 16 > sl.cap,
                Phase::Decode { .. } => sl.full && a.pos + 9 >= sl.cap,
            };
            if stuck {
                let a = sl.active.take().unwrap();
                sl.last_used = Instant::now();
                match a.phase {
                    Phase::Prefill { .. } => {
                        let _ = a.job.tx.send(Event::Failed(format!("{NO_MEMORY}: this request needs about {} tokens of context and the GPU has room for {} in this slot right now", a.job.prompt.len() + 16, sl.cap)));
                        // Nothing of it is worth keeping: hand back whatever the slot holds beyond its previous conversation.
                        eng.trim(i, sl.hist.len() + TMAX + 16);
                        sl.cap = eng.ctx_mapped(i);
                    }
                    Phase::Decode { .. } => finish(a, "length"),
                }
            }
        }

        // ---- plan the step: generating conversations first, then prompt chunks with what is left
        let decoders = slots.iter().filter_map(|s| s.active.as_ref()).filter(|a| matches!(a.phase, Phase::Decode { .. })).count();
        let prefillers = slots.iter().filter_map(|s| s.active.as_ref()).filter(|a| matches!(a.phase, Phase::Prefill { .. })).count();
        let (k_max, tau) = if cfg.tau.is_some() || multi.is_some() { fixed_k(&cfg, decoders, multi) } else { (cfg.k_max, 0.0) };
        let mut rows: Vec<(usize, usize, u32)> = Vec::new();
        let mut excl: Vec<i32> = Vec::new();
        let mut sampling: Vec<(f32, u32)> = Vec::new();
        let mut plan: Vec<Option<Plan>> = (0..nslots).map(|_| None).collect();
        let mut qrows: Vec<i32> = Vec::new();
        // Drafts for every conversation that will speculate, computed together: the drafter reads its weights once
        // per pass, so a pass for four conversations costs little more than a pass for one.
        let mut want: Vec<(usize, u32, usize)> = Vec::new(); // (slot, anchor, anchor position)
        let mut k_cap = vec![0usize; nslots];
        let mut copied: Vec<Option<(Vec<u32>, usize)>> = (0..nslots).map(|_| None).collect();
        if k_max > 0 && eng.has_draft() {
            for (s, slot) in slots.iter_mut().enumerate() {
                let cap = slot.cap;
                let Some(a) = slot.active.as_mut() else { continue };
                let Phase::Decode { anchor } = a.phase else { continue };
                k_cap[s] = if a.forced.is_empty() { k_max.min(7).min(cap.saturating_sub(a.pos + 8)) } else { 0 };
                // A conversation that is repeating an earlier passage drafts by copying, and skips the drafter.
                if k_cap[s] > 0 && cfg.copy_max > 0 && a.copy.run >= 4 {
                    let room = cfg.copy_max.min(Engine::max_spec_run() - 1).min(cap.saturating_sub(a.pos + 9));
                    if a.copy.worth(room, a.draft_yield) {
                        copied[s] = Some((a.copy.drafts(room), a.copy.run));
                        continue;
                    }
                    a.copy.forget();
                }
                if k_cap[s] > 0 {
                    want.push((s, anchor, a.pos));
                }
            }
        }
        let mut drafted: Vec<Option<Drafts>> = vec![None; nslots];
        let t_draft = Instant::now();
        for group in want.chunks(draft_batch) {
            let n = group.iter().map(|w| k_cap[w.0]).max().unwrap_or(1);
            // Conversations that sample get drafts sampled from the drafter's distribution.
            let how: Vec<(f32, u32)> = group.iter().map(|w| slots[w.0].active.as_ref().map_or((0.0, 0), |a| (if sampled_drafts(a.job.temperature) { a.job.temperature } else { 0.0 }, a.job.seed))).collect();
            if let Some(res) = eng.draft_batch(group, n, Some(&how)) {
                for (w, r) in group.iter().zip(res) {
                    drafted[w.0] = Some(r);
                }
            }
        }
        if !want.is_empty() {
            spec.draft_ms = spec.draft_ms * 0.95 + 0.05 * t_draft.elapsed().as_secs_f32() * 1e3;
        }
        // How many of them to verify.
        let is_sampled = |s: usize| slots[s].active.as_ref().map_or(false, |a| sampled_drafts(a.job.temperature));
        let k_take: Vec<usize> = if cfg.tau.is_some() || multi.is_some() {
            (0..nslots).map(|s| match (&copied[s], drafted[s]) {
                // copied tokens: as many as are all accepted together with probability >= tau
                (Some((d, run)), _) => {
                    let p = slots[s].active.as_ref().map_or(0.0, |a| a.copy.accept(*run));
                    (1..=d.len()).take_while(|&j| p.powi(j as i32) >= tau).count()
                }
                (None, Some(d)) => d.top[..k_cap[s]].iter().take_while(|&&c| c >= tau).count(),
                _ => 0,
            }).collect()
        } else {
            let list: Vec<(usize, Vec<f32>)> = (0..nslots).filter_map(|s| match (&copied[s], drafted[s]) {
                (Some((d, run)), _) => Some((s, vec![slots[s].active.as_ref().map_or(0.0, |a| a.copy.accept(*run)); d.len()])),
                (None, Some(d)) => Some((s, d.top[..k_cap[s]].iter().map(|&t| spec.accept(t, is_sampled(s))).collect())),
                _ => None,
            }).collect();
            let forced_rows: usize = slots.iter().filter_map(|sl| sl.active.as_ref()).map(|a| a.forced.len()).sum();
            let prompt_rows = if prefillers > 0 { prefill_share.min(TMAX - decoders.min(TMAX)) } else { 0 };
            spec.choose(&list, decoders + forced_rows + prompt_rows, decoders, prompt_rows, nslots, !want.is_empty())
        };
        // Text the server must supply (the opening of a tool call) has first claim on the step's rows: drafts of
        // earlier slots may not crowd it out.
        let forced_len: Vec<usize> = slots.iter().map(|sl| sl.active.as_ref().map_or(0, |a| if matches!(a.phase, Phase::Decode { .. }) { a.forced.len() } else { 0 })).collect();
        for (s, slot) in slots.iter_mut().enumerate() {
            let lim = slot.cap;
            let Some(a) = slot.active.as_mut() else { continue };
            let Phase::Decode { anchor } = a.phase else { continue };
            let free_rows = TMAX.saturating_sub(rows.len() + 1 + (nslots - s) + forced_len[s + 1..].iter().sum::<usize>());
            let (mut fed, mut confs, mut tops, mut sampled, mut copy_run) = (Vec::new(), [0f32; 7], [0f32; 7], false, None);
            let forcing = !a.forced.is_empty() && a.forced.len() <= free_rows.min(lim.saturating_sub(a.pos + 8));
            if forcing {
                fed = std::mem::take(&mut a.forced);
            } else {
                a.forced.clear();
                if let Some((d, run)) = &copied[s] {
                    fed.extend_from_slice(&d[..k_take[s].min(free_rows).min(d.len())]);
                    copy_run = Some(*run);
                } else if let Some(d) = drafted[s] {
                    fed.extend_from_slice(&d.tokens[..k_take[s].min(free_rows)]);
                    (confs, tops, sampled) = (d.conf, d.top, sampled_drafts(a.job.temperature));
                }
            }
            let r0 = rows.len();
            rows.push((s, a.pos, anchor));
            for (i, &t) in fed.iter().enumerate() {
                rows.push((s, a.pos + 1 + i, t));
                excl.push(if forcing { -1 } else { t as i32 });
                qrows.push(if sampled && !forcing { (s * 7 + i) as i32 } else { -1 }); // the row whose prediction is checked against draft i
            }
            excl.push(-1);
            qrows.push(-1);
            sampling.extend(std::iter::repeat((a.job.temperature, a.job.seed)).take(fed.len() + 1));
            plan[s] = Some(Plan { r0, n: fed.len() + 1, fed, forcing, confs, tops, sampled, copy_run });
        }
        if prefillers > 0 {
            let budget = (TMAX - rows.len()).min(if decoders > 0 { prefill_share } else { TMAX });
            let each = (budget / prefillers).max(1);
            for (s, slot) in slots.iter_mut().enumerate() {
                let Some(a) = slot.active.as_ref() else { continue };
                let Phase::Prefill { done } = a.phase else { continue };
                let len = a.job.prompt.len();
                let stop = if prefix_cache && done < snapshot_at(len) { snapshot_at(len) } else { len };
                let c = each.min(stop - done).min(TMAX - rows.len()).min(slot.cap.saturating_sub(a.pos));
                if c == 0 {
                    continue;
                }
                plan[s] = Some(Plan { r0: rows.len(), n: c, fed: Vec::new(), forcing: false, confs: [0.0; 7], tops: [0.0; 7], sampled: false, copy_run: None });
                for i in 0..c {
                    rows.push((s, a.pos + i, a.job.prompt[done + i]));
                    excl.push(-1);
                    qrows.push(-1);
                    sampling.push((a.job.temperature, a.job.seed));
                }
            }
        }
        if rows.is_empty() {
            continue;
        }
        // The output head is only needed if some row's prediction will be used.
        let head = decoders > 0 || slots.iter().enumerate().any(|(s, sl)| match (&sl.active, &plan[s]) {
            (Some(a), Some(p)) => matches!(a.phase, Phase::Prefill { done } if done + p.n == a.job.prompt.len()),
            _ => false,
        });
        let t_step = Instant::now();
        let res = eng.step_rows_with(&rows, Some(&excl), head, Some(&sampling), Some(&qrows));
        spec.observe_step(rows.len(), t_step.elapsed().as_secs_f32() * 1e3);

        // ---- hand out the results
        for s in 0..nslots {
            let Some(Plan { r0, n: n_rows, fed: drafts, forcing, confs, tops, sampled: sampled_draft, copy_run }) = plan[s].take() else { continue };
            let mut a = slots[s].active.take().unwrap();
            let sampled = a.job.temperature > 0.0;
            let mut out: Vec<u32> = Vec::new();
            match a.phase {
                Phase::Prefill { done } => {
                    let (end, len) = (done + n_rows, a.job.prompt.len());
                    a.pos += n_rows;
                    slots[s].hist.extend_from_slice(&a.job.prompt[done..end]);
                    slots[s].hi = slots[s].hi.max(a.pos);
                    if prefix_cache && end == snapshot_at(len) {
                        // The next request of this conversation will most likely extend (or repeat) this prompt.
                        eng.commit(s, r0 + n_rows - 1);
                        slots[s].ckpt = eng.snapshot(s).then_some(a.pos);
                        slots[s].ckpt_hi = slots[s].hi;
                    }
                    if end == len {
                        let r = res[r0 + n_rows - 1];
                        out.push(if sampled { r.sample } else { r.greedy });
                        if force_tools {
                            watch_tool_call(&mut a, &out, chat);
                        }
                        a.ttft = a.job.t0.elapsed().as_secs_f64();
                        eng.commit(s, r0 + n_rows - 1);
                    } else {
                        a.phase = Phase::Prefill { done: end };
                    }
                }
                Phase::Decode { anchor } => {
                    let k = n_rows - 1;
                    let (mut n, mut last) = (0usize, None);
                    if forcing {
                        n = k; // supplied tokens are taken as they are
                    } else if sampled && sampled_draft {
                        // Drafts drawn from the drafter's distribution q: kept with probability min(1, p / q); a rejected
                        // position takes the row's sample from the leftover distribution.
                        while n < k {
                            let p = res[r0 + n].logp_excl.exp();
                            if (a.rng.uniform() as f32) < (p / confs[n].max(1e-30)).min(1.0) {
                                n += 1;
                            } else {
                                last = Some(res[r0 + n].sample_excl);
                                break;
                            }
                        }
                    } else if sampled {
                        // A draft is kept exactly when it is the token the sampler draws at that position anyway. That
                        // happens with probability p(draft), and it makes the output depend only on the seed: the
                        // same text with any number of drafts verified, or none.
                        while n < k && res[r0 + n].sample == drafts[n] {
                            n += 1;
                        }
                    } else {
                        while n < k && res[r0 + n].greedy == drafts[n] {
                            n += 1;
                        }
                    }
                    if let Some(run) = copy_run {
                        for i in 0..(n + 1).min(k) {
                            a.copy.observe(run, i < n);
                        }
                        a.copied += n;
                    } else if !forcing {
                        a.draft_yield = 0.9 * a.draft_yield + 0.1 * n as f32;
                        for i in 0..(n + 1).min(k) {
                            spec.observe(tops[i], sampled_draft, i < n); // drafts 0..n were accepted, draft n (if any) was not
                        }
                    }
                    out.extend_from_slice(&drafts[..n]);
                    out.push(last.unwrap_or(if sampled { res[r0 + n].sample } else { res[r0 + n].greedy }));
                    if force_tools {
                        // A tool call opened inside this step: what follows its opening token is replaced by the fixed text.
                        let keep = watch_tool_call(&mut a, &out, chat);
                        out.truncate(keep);
                        n = keep - 1;
                    }
                    slots[s].hi = slots[s].hi.max(a.pos + n_rows); // written: the anchor and every fed token, kept or not
                    slots[s].hist.push(anchor); // positions now held by the caches: the anchor and the accepted tokens
                    slots[s].hist.extend_from_slice(&drafts[..n]);
                    a.pos += n + 1; // rejected positions are simply overwritten later
                    a.stats.steps += 1;
                    if !forcing {
                        a.stats.verified += k;
                        a.stats.accepted += n;
                    }
                    eng.commit(s, r0 + n);
                }
            }
            if out.is_empty() {
                slots[s].active = Some(a);
                continue;
            }
            // Count and cut at the limits, then pass the tokens on.
            let mut send = Vec::with_capacity(out.len());
            let mut why = None;
            for t in out {
                if a.n_out >= a.job.max_tokens {
                    why = Some("length");
                    break;
                }
                a.n_out += 1;
                send.push(t);
                if t == EOS {
                    why = Some("stop");
                    break;
                }
            }
            if why.is_none() && a.n_out >= a.job.max_tokens {
                why = Some("length");
            }
            if why.is_none() && (a.pos + 9 >= max_ctx || (slots[s].full && a.pos + 9 >= slots[s].cap)) {
                why = Some("length");
            }
            let last = *send.last().unwrap();
            if cfg.copy_max > 0 {
                for &t in &send {
                    a.copy.push(t);
                }
            }
            let alive = a.job.tx.send(Event::Tokens(send)).is_ok();
            slots[s].last_used = Instant::now();
            match why {
                Some(w) => finish(a, w),
                None if !alive => finish(a, "cancelled"), // client went away: free the slot
                None => {
                    a.phase = Phase::Decode { anchor: last };
                    slots[s].active = Some(a);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------- HTTP side
/// One header line, at most 8 KiB.
fn read_line(r: &mut BufReader<&TcpStream>) -> Result<String> {
    const MAX_LINE: u64 = 8192;
    let mut line = Vec::new();
    let n = r.by_ref().take(MAX_LINE).read_until(b'\n', &mut line)?;
    if n as u64 == MAX_LINE && !line.ends_with(b"\n") {
        bail!("header line too long");
    }
    Ok(String::from_utf8_lossy(&line).into_owned())
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
    /// Access-Control-Request-Headers of a CORS preflight.
    cors_headers: Option<String>,
}

fn read_request(mut stream: &TcpStream) -> Result<Request> {
    let t0 = Instant::now();
    let mut r = BufReader::new(stream);
    let line = read_line(&mut r)?;
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or(""));
    let path = target.split(['?', '#']).next().unwrap_or("").to_string();
    let (mut len, mut expect, mut cors_headers) = (0usize, false, None);
    for n in 0.. {
        let h = read_line(&mut r)?;
        if h.is_empty() || h == "\r\n" || h == "\n" {
            break;
        }
        if n >= 128 || t0.elapsed() > Duration::from_secs(60) {
            bail!("too many header lines, or headers arriving too slowly");
        }
        if let Some((k, v)) = h.split_once(':') {
            let v = v.trim();
            if k.eq_ignore_ascii_case("content-length") {
                len = v.parse().map_err(|_| anyhow!("bad Content-Length"))?;
            } else if k.eq_ignore_ascii_case("transfer-encoding") {
                bail!("chunked request bodies are not supported; send Content-Length");
            } else if k.eq_ignore_ascii_case("expect") {
                expect = v.eq_ignore_ascii_case("100-continue");
            } else if k.eq_ignore_ascii_case("access-control-request-headers") {
                cors_headers = Some(v.to_string());
            }
        }
    }
    if len > MAX_BODY {
        bail!("{TOO_LARGE}: {len} bytes, the limit is {MAX_BODY}");
    }
    if expect && len > 0 {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?; // curl waits a second for this before sending a large body
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, body, cors_headers })
}
const TOO_LARGE: &str = "request body too large";
const NO_MEMORY: &str = "out of GPU memory";

/// True once the client has closed its side of the connection.
fn client_gone(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return true;
    }
    let r = stream.peek(&mut [0u8; 1]);
    let _ = stream.set_nonblocking(false);
    match r {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
    }
}

/// `--cors`: the origin browsers are told may call this server (none by default).
static CORS: OnceLock<Option<String>> = OnceLock::new();
fn cors_header() -> String {
    CORS.get().and_then(|c| c.as_ref()).map_or(String::new(), |o| format!("Access-Control-Allow-Origin: {o}\r\n"))
}

/// One write per response: with TCP_NODELAY every write is a packet.
fn respond(mut s: &TcpStream, status: &str, body: &Value) -> Result<()> {
    let b = serde_json::to_vec(body)?;
    let mut out = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n", b.len(), cors_header()).into_bytes();
    out.extend_from_slice(&b);
    s.write_all(&out)?;
    Ok(())
}

/// Holds back text that might be the beginning of a stop sequence until it is known not to be one.
struct StopFilter {
    stops: Vec<String>,
    held: String,
}
impl StopFilter {
    /// Takes newly generated text; returns the text that may be shown and whether a stop sequence was reached.
    fn push(&mut self, piece: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (piece.to_string(), false);
        }
        self.held.push_str(piece);
        if let Some(i) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
            self.held.truncate(i);
            return (std::mem::take(&mut self.held), true);
        }
        // Keep the longest tail that is a proper prefix of a stop sequence.
        let mut keep = 0;
        for s in &self.stops {
            let longest = (1..s.len().min(self.held.len() + 1)).rev().find(|&k| s.is_char_boundary(k) && self.held.ends_with(&s[..k]));
            keep = keep.max(longest.unwrap_or(0));
        }
        let cut = self.held.len() - keep;
        (self.held.drain(..cut).collect(), false)
    }
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod stop_tests {
    use super::StopFilter;

    /// Feeds `text` in pieces of `n` bytes; returns what was shown and whether a stop sequence ended it.
    fn run(stops: &[&str], text: &str, n: usize) -> (String, bool) {
        let mut f = StopFilter { stops: stops.iter().map(|s| s.to_string()).collect(), held: String::new() };
        let mut shown = String::new();
        let chars: Vec<char> = text.chars().collect();
        for piece in chars.chunks(n) {
            let (t, hit) = f.push(&piece.iter().collect::<String>());
            shown.push_str(&t);
            if hit {
                return (shown, true);
            }
        }
        shown.push_str(&f.finish());
        (shown, false)
    }

    #[test]
    fn stop_sequences_cut_the_text_however_it_is_split() {
        for n in 1..12 {
            assert_eq!(run(&["\nObservation:"], "Thought: look\nAction: read\nObservation: x", n), ("Thought: look\nAction: read".to_string(), true));
            assert_eq!(run(&["END", "\n\n"], "one\ntwo\n\nthree END", n), ("one\ntwo".to_string(), true));
            // A beginning of a stop sequence that does not complete is shown after all.
            assert_eq!(run(&["\nObservation:"], "a\nObserver\nObs", n), ("a\nObserver\nObs".to_string(), false));
            assert_eq!(run(&["é!"], "caf\u{e9} \u{e9}? \u{e9}!", n), ("caf\u{e9} \u{e9}? ".to_string(), true));
            assert_eq!(run(&[], "plain", n), ("plain".to_string(), false));
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    None,
    Reasoning,
    Content,
    Tool,
}

struct Shared {
    chat: Chat,
    model_name: String,
    max_ctx: usize,
    slots: usize,
    jobs: mpsc::Sender<Job>,
}

fn handle(stream: TcpStream, sh: Arc<Shared>) {
    let _ = stream.set_nodelay(true);
    // A silent or stalled client must not hold a thread forever.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let mut sse_started = false;
    let result = (|| -> Result<()> {
        let req = match read_request(&stream) {
            Ok(r) => r,
            Err(e) if e.downcast_ref::<std::io::Error>().is_none() => {
                let too_large = e.to_string().starts_with(TOO_LARGE);
                respond(&stream, if too_large { "413 Payload Too Large" } else { "400 Bad Request" }, &json!({"error": {"message": e.to_string()}}))?;
                if too_large {
                    // Closing with the body unread resets the connection and the client never sees the 413:
                    // let it finish sending (bounded), then close.
                    let _ = stream.shutdown(Shutdown::Write);
                    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                    let (t0, mut sink, mut total) = (Instant::now(), vec![0u8; 1 << 16], 0usize);
                    while let Ok(n) = (&stream).read(&mut sink) {
                        total += n;
                        if n == 0 || total > 8 * MAX_BODY || t0.elapsed() > Duration::from_secs(10) {
                            break;
                        }
                    }
                }
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => respond(&stream, "200 OK", &json!({"object": "list", "data": [{"id": sh.model_name, "object": "model", "owned_by": "local",
                "context_window": sh.max_ctx, "max_model_len": sh.max_ctx, "slots": sh.slots}]})),
            ("GET", "/health") => respond(&stream, "200 OK", &json!({"status": "ok"})),
            ("POST", "/v1/chat/completions") => match completion(&stream, &req.body, &sh, &mut sse_started) {
                Err(e) if e.downcast_ref::<std::io::Error>().is_none() => {
                    if sse_started {
                        // The 200 and some events are already on the wire: report inside the stream, then end it.
                        (&stream).write_all(format!("data: {}\n\ndata: [DONE]\n\n", json!({"error": {"message": e.to_string()}})).as_bytes())?;
                        Ok(())
                    } else {
                        let status = if e.to_string().starts_with(NO_MEMORY) { "503 Service Unavailable" } else { "400 Bad Request" };
                        respond(&stream, status, &json!({"error": {"message": e.to_string()}}))
                    }
                }
                other => other,
            },
            // A browser asks before it posts from another origin. Answered only when --cors names who may.
            ("OPTIONS", _) if CORS.get().map_or(false, |c| c.is_some()) => {
                let allow = req.cors_headers.unwrap_or_else(|| "Authorization, Content-Type".into());
                (&stream).write_all(format!("HTTP/1.1 204 No Content\r\n{}Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: {allow}\r\nAccess-Control-Max-Age: 600\r\nConnection: close\r\n\r\n", cors_header()).as_bytes())?;
                Ok(())
            }
            _ => respond(&stream, "404 Not Found", &json!({"error": {"message": "not found"}})),
        }
    })();
    if let Err(e) = result {
        eprintln!("request failed: {e}");
    }
}

fn completion(mut stream: &TcpStream, body: &[u8], sh: &Shared, sse_started: &mut bool) -> Result<()> {
    let t0 = Instant::now();
    let chat = &sh.chat;
    let req: Value = serde_json::from_slice(body)?;
    let messages = req.get("messages").filter(|m| m.as_array().map_or(false, |a| !a.is_empty())).ok_or_else(|| anyhow!("messages must be a non-empty array"))?;
    let effort = req.get("reasoning_effort").or_else(|| req.pointer("/chat_template_kwargs/reasoning_effort")).filter(|v| !v.is_null());
    // tool_choice "none": the model is not shown the tools at all.
    let tools = req.get("tools").filter(|v| !v.is_null() && req.get("tool_choice").and_then(Value::as_str) != Some("none"));
    // Fields that would change the reply and are not implemented are refused, not ignored: a client that asked for
    // JSON or for log-probabilities would otherwise fail later, somewhere less obvious.
    if req.get("n").filter(|v| !v.is_null()).map_or(false, |n| n.as_u64() != Some(1)) {
        bail!("n must be 1");
    }
    if req.get("logprobs").and_then(Value::as_bool) == Some(true) || req.get("top_logprobs").and_then(Value::as_u64).unwrap_or(0) > 0 {
        bail!("logprobs are not supported");
    }
    if let Some(format) = req.get("response_format").filter(|v| !v.is_null()) {
        if format.get("type").and_then(Value::as_str) != Some("text") {
            bail!("response_format is not supported: replies are plain text");
        }
    }
    let stops: Vec<String> = match req.get("stop").filter(|v| !v.is_null()) {
        None => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.len() <= 16 && a.iter().all(Value::is_string) => a.iter().map(|s| s.as_str().unwrap().to_string()).collect(),
        Some(_) => bail!("stop must be a string or a list of up to 16 strings"),
    };
    if stops.iter().any(|s| s.len() > 256) {
        bail!("a stop sequence can be at most 256 bytes");
    }
    // Rendering the prompt takes time that grows faster than the number of tools and tool calls: bound them.
    if tools.map_or(false, |t| !t.as_array().map_or(false, |a| a.len() <= 512 && a.iter().all(Value::is_object))) {
        bail!("tools must be a list of at most 512 tool definitions");
    }
    if messages.as_array().map_or(0, |a| a.len()) > 8192 {
        bail!("a conversation can have at most 8192 messages");
    }
    let mut stop = StopFilter { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new() };
    let prompt = chat.encode(&chat.render(messages, tools, effort)?)?;
    if prompt.is_empty() || prompt.len() + 17 >= sh.max_ctx {
        bail!("prompt has {} tokens; this server holds {} tokens of context per conversation", prompt.len(), sh.max_ctx);
    }
    let room = sh.max_ctx - prompt.len() - 16; // headroom for one speculative batch
    let max_tokens = match req.get("max_completion_tokens").or_else(|| req.get("max_tokens")).filter(|v| !v.is_null()) {
        None => room,
        Some(v) => (v.as_u64().filter(|&n| n > 0).ok_or_else(|| anyhow!("max_tokens must be a positive integer"))? as usize).min(room),
    };
    let temperature = match req.get("temperature").filter(|v| !v.is_null()) {
        None => 1.0,
        Some(v) => v.as_f64().filter(|t| (0.0..=2.0).contains(t)).ok_or_else(|| anyhow!("temperature must be a number between 0 and 2"))?,
    };
    let temperature = if temperature < 0.01 { 0.0 } else { temperature }; // below that it is greedy decoding in all but name
    let streaming = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let seed = match req.get("seed").filter(|v| !v.is_null()) {
        None => now.as_nanos() as u32,
        Some(v) if v.as_i64() == Some(-1) => now.as_nanos() as u32, // llama.cpp's spelling of "random"
        Some(v) => v.as_u64().ok_or_else(|| anyhow!("seed must be a non-negative integer (or -1 for random)"))? as u32,
    };
    let id = format!("chatcmpl-{:x}", now.as_nanos() as u64 ^ seed as u64);
    let created = now.as_secs();
    let prompt_tokens = prompt.len();
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    // Leaving this function for any reason (any `?` below) tells the scheduler to drop the job and free the slot.
    let _cancel_on_exit = CancelOnDrop(cancel.clone());
    sh.jobs.send(Job { prompt, max_tokens, temperature: temperature as f32, seed, tx, t0, cancel: cancel.clone() }).map_err(|_| anyhow!("engine stopped"))?;

    let chunk = |delta: Value, finish: Value, usage: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": sh.model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage});
    // Bytes for a streaming client, sent once per batch of tokens. The response header goes out with the first
    // event, so a request that fails before producing anything can still be answered with a proper status.
    let mut out: Vec<u8> = Vec::new();
    macro_rules! sse {
        ($event:expr) => {
            if streaming {
                if !*sse_started {
                    *sse_started = true;
                    out.extend_from_slice(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{}Connection: close\r\n\r\n", cors_header()).as_bytes());
                    out.extend_from_slice(format!("data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), Value::Null, Value::Null)).as_bytes());
                }
                out.extend_from_slice(format!("data: {}\n\n", $event).as_bytes());
            }
        };
    }

    let (mut mode, mut seg) = (Mode::None, Vec::<u32>::new());
    let (mut content, mut reasoning, mut tool_text) = (String::new(), String::new(), String::new());
    let mut tool_calls: Vec<Value> = Vec::new();
    let (mut finish, mut n_out, mut cached, mut stopped) = ("length", 0usize, 0usize, false);
    let debug_tokens = std::env::var("INK_DEBUG_TOKENS").is_ok();
    // Text of the answer: shown once it cannot be the start of a stop sequence.
    macro_rules! say {
        ($text:expr) => {{
            let text: String = $text;
            if !text.is_empty() {
                sse!(chunk(json!({"content": text}), Value::Null, Value::Null));
                content.push_str(&text);
            }
        }};
    }
    // Ends a message. A tool call's {"name", "args"} becomes an OpenAI tool call; text that is not valid JSON is
    // handed over as content rather than dropped.
    macro_rules! end_message {
        () => {
            if mode == Mode::Content {
                say!(stop.finish());
            }
            if mode == Mode::Tool && !tool_text.trim().is_empty() {
                if let Ok(v) = serde_json::from_str::<Value>(tool_text.trim()) {
                    let call = json!({"index": tool_calls.len(), "id": format!("call_{:x}{:04x}", now.as_nanos() as u64, CALL_SEQ.fetch_add(1, Ordering::Relaxed) & 0xffff), "type": "function",
                        "function": {"name": v.get("name").cloned().unwrap_or(Value::Null), "arguments": v.get("args").or_else(|| v.get("arguments")).map(|a| a.to_string()).unwrap_or_else(|| "{}".into())}});
                    sse!(chunk(json!({"tool_calls": [call.clone()]}), Value::Null, Value::Null));
                    tool_calls.push(call);
                } else {
                    say!(std::mem::take(&mut tool_text));
                }
            }
            tool_text.clear();
        };
    }
    let mut last_look = Instant::now();
    'events: loop {
        // A client that has closed its connection is not waiting for an answer. Checked on the clock, not only
        // when the engine is quiet: a non-streaming reply writes nothing until it is complete.
        if last_look.elapsed() >= Duration::from_millis(250) {
            last_look = Instant::now();
            if client_gone(stream) {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "client disconnected").into());
            }
        }
        let ev = match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(ev) => ev,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let toks = match ev {
            Event::Tokens(t) => t,
            Event::Failed(msg) => bail!(msg),
            Event::Done { finish: f, completion_tokens, cached: c, summary } => {
                finish = if stopped { "stop" } else { f };
                n_out = completion_tokens;
                cached = c;
                eprintln!("request: {}", if stopped { summary.replace("finish=cancelled", "finish=stop (stop sequence)") } else { summary });
                break;
            }
        };
        if stopped {
            continue; // generated before the scheduler saw the cancellation
        }
        if debug_tokens {
            eprintln!("tokens: {:?}", toks.iter().map(|&t| (t, chat.decode(&[t]).unwrap_or_default())).collect::<Vec<_>>());
        }
        for tok in toks {
            if tok == EOS {
                continue;
            }
            if tok >= FIRST_SPECIAL {
                // Control token: switches between "thinking", "answer", "tool call" and "between messages".
                end_message!();
                mode = match tok {
                    T_THINK => Mode::Reasoning,
                    T_TOOL_JSON => Mode::Tool,
                    T_END | 200_000..=200_003 => Mode::None,
                    _ => Mode::Content,
                };
                seg.clear();
            } else if mode != Mode::None {
                // Decode only the tokens not yet shown. A token can end in the middle of a UTF-8 character; in that
                // case keep it pending until the character completes.
                seg.push(tok);
                let piece = chat.decode(&seg)?;
                if piece.ends_with('\u{FFFD}') {
                    continue;
                }
                seg.clear();
                match mode {
                    Mode::Tool => tool_text.push_str(&piece),
                    Mode::Reasoning => {
                        if !piece.is_empty() {
                            sse!(chunk(json!({"reasoning_content": piece}), Value::Null, Value::Null));
                        }
                        reasoning.push_str(&piece);
                    }
                    _ => {
                        let (text, hit) = stop.push(&piece);
                        say!(text);
                        if hit {
                            // The reply ends here. Stop generating, then wait for the scheduler's account of it.
                            stopped = true;
                            cancel.store(true, Ordering::Relaxed);
                            continue 'events;
                        }
                    }
                }
            }
        }
        if !out.is_empty() {
            stream.write_all(&out)?;
            out.clear();
        }
    }
    if stopped {
        finish = "stop";
    }
    // The stream can end inside a message (token limit, or end of turn without <|end_message|>).
    if !seg.is_empty() && mode != Mode::None && !stopped {
        let piece = chat.decode(&seg)?.trim_end_matches('\u{FFFD}').to_string();
        match mode {
            Mode::Tool => tool_text.push_str(&piece),
            Mode::Reasoning => {
                if !piece.is_empty() {
                    sse!(chunk(json!({"reasoning_content": piece}), Value::Null, Value::Null));
                }
                reasoning.push_str(&piece);
            }
            _ => say!(stop.push(&piece).0),
        }
    }
    if !stopped {
        end_message!();
    }
    let usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": n_out, "total_tokens": prompt_tokens + n_out,
        "prompt_tokens_details": {"cached_tokens": cached}});
    if !tool_calls.is_empty() && finish == "stop" {
        finish = "tool_calls";
    }
    if streaming {
        sse!(chunk(json!({}), json!(finish), usage));
        out.extend_from_slice(b"data: [DONE]\n\n");
        stream.write_all(&out)?;
        stream.flush()?;
    } else {
        let mut msg = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = json!(tool_calls);
        }
        respond(stream, "200 OK", &json!({"id": id, "object": "chat.completion", "created": created, "model": sh.model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}))?;
    }
    Ok(())
}

pub fn serve(addr: &str, model_name: &str, chat: Chat, eng: &mut Engine, max_ctx: usize, cfg: Options) -> Result<()> {
    let listener = TcpListener::bind(addr)?;
    let _ = CORS.set(cfg.cors.clone());
    eprintln!(
        "listening on http://{addr}/v1  (model id: {model_name}; {} slot(s), a conversation can reach {max_ctx} tokens; speculation: up to {} drafts/step, {})",
        eng.slots(), cfg.k_max, cfg.tau.map_or("chosen by expected gain".to_string(), |t| format!("confidence >= {t}"))
    );
    let (jobs, rx) = mpsc::channel();
    let sh = Arc::new(Shared { chat, model_name: model_name.to_string(), max_ctx, slots: eng.slots(), jobs });
    let sh2 = sh.clone();
    std::thread::spawn(move || {
        static OPEN: AtomicUsize = AtomicUsize::new(0);
        for stream in listener.incoming() {
            let Ok(stream) = stream else {
                std::thread::sleep(Duration::from_millis(50)); // e.g. out of file descriptors: do not spin
                continue;
            };
            if OPEN.fetch_add(1, Ordering::Relaxed) >= MAX_CONNECTIONS {
                OPEN.fetch_sub(1, Ordering::Relaxed);
                let _ = respond(&stream, "503 Service Unavailable", &json!({"error": {"message": "too many connections"}}));
                continue;
            }
            let sh = sh2.clone();
            std::thread::spawn(move || {
                handle(stream, sh);
                OPEN.fetch_sub(1, Ordering::Relaxed);
            });
        }
    });
    scheduler(eng, rx, cfg, max_ctx, &sh.chat);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_follows_a_repeated_passage() {
        // 1..=10, some filler, then the conversation starts repeating 3 4 5
        let mut c = Copy::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52]);
        assert!(c.drafts(8).is_empty());
        for t in [3, 4] {
            c.push(t);
        }
        assert!(c.drafts(8).is_empty()); // two tokens are not enough
        c.push(5);
        assert_eq!((c.run, c.drafts(4)), (3, vec![6u32, 7, 8, 9]));
        assert!(!c.worth(24, 0.0), "a match of three tokens is not used");
        c.push(6);
        c.push(7);
        assert_eq!((c.run, c.drafts(3)), (5, vec![8u32, 9, 10]));
        assert!(c.worth(24, 1.0) && !c.worth(24, 6.0), "copying has to beat what the drafter yields");
        c.push(99); // the model went its own way
        assert_eq!(c.run, 0);
        assert!(c.drafts(8).is_empty() && !c.worth(24, 0.0));
    }

    #[test]
    fn copy_continues_a_repeating_pattern_into_itself() {
        let mut c = Copy::new(&[7, 8, 9, 7, 8, 9]);
        // the last three tokens match positions 0..3; what followed there is 7 8 9, and then the pattern again
        assert_eq!(c.drafts(8), vec![7u32, 8, 9, 7, 8, 9, 7, 8]);
        c.push(7);
        assert_eq!(c.drafts(2), vec![8u32, 9]);
        // a run of one token, and an alternation: as many drafts as asked for, not one or two
        let ones = Copy::new(&[5, 4, 4, 4, 4, 4]);
        assert_eq!(ones.drafts(6), vec![4u32; 6]);
        let two = Copy::new(&[1, 2, 1, 2, 1, 2, 1]);
        assert_eq!(two.drafts(5), vec![2u32, 1, 2, 1, 2]);
        assert!(Copy::new(&[]).drafts(4).is_empty() && c.drafts(0).is_empty());
    }

    #[test]
    fn copy_matches_are_real_and_drafts_agree_with_brute_force() {
        // pseudo-random tokens from a tiny vocabulary: many accidental matches
        let mut x = 12345u32;
        let mut c = Copy::new(&[]);
        for _ in 0..3000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            c.push((x >> 24) % 5);
            let i = c.seq.len() - 1;
            if let Some(s) = c.src {
                assert!(s < i);
                assert!(c.run >= 1 && (0..c.run.min(s + 1)).all(|n| c.seq[s - n] == c.seq[i - n]), "claimed match of {} at {s}", c.run);
                let d = c.drafts(7);
                assert_eq!(d.len(), 7);
                let mut ext = c.seq.clone();
                for (j, &t) in d.iter().enumerate() {
                    assert_eq!(t, ext[s + 1 + j]); // each draft is what followed the passage, the drafts included
                    ext.push(t);
                }
            }
        }
    }

    #[test]
    fn copy_calibration_switches_off_and_comes_back() {
        let mut c = Copy::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(c.run, 10);
        assert!(c.worth(24, 3.0));
        for _ in 0..12 {
            c.observe(c.run, false); // every copied token of this length is rejected
        }
        assert!(c.accept(c.run) < 0.4 && !c.worth(24, 3.0));
        for _ in 0..60 {
            c.forget(); // not used for a while: the bad record fades
        }
        assert!(c.worth(24, 3.0), "copying must be tried again eventually");
        // a good record makes long copies worth more than a strong drafter
        for _ in 0..50 {
            c.observe(c.run, true);
        }
        assert!(c.accept(c.run) > 0.95 && c.worth(24, 6.0));
    }

    #[test]
    fn resume_needs_the_conv_history_before_the_restart_point() {
        let hist: Vec<u32> = (0..100).collect();
        let slot = |hi: usize| Slot { hist: hist.clone(), ckpt: Some(40), hi, ckpt_hi: 0, cap: 4096, full: false, last_used: Instant::now(), active: None };
        let prompt: Vec<u32> = (0..120).collect();
        let margin = Engine::max_spec_run() - 3;
        assert_eq!(resume_point(&slot(100 + margin), &prompt), (100, false)); // live: rejected rows written up to the margin
        assert_eq!(resume_point(&slot(100 + margin + 1), &prompt), (40, true)); // one more: back to the snapshot
        let other: Vec<u32> = (500..520).collect();
        assert_eq!(resume_point(&slot(100), &other), (0, false)); // nothing shared
    }

    #[test]
    fn copy_measures_the_match_length_backwards() {
        let mut c = Copy::new(&[1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3, 4]);
        assert_eq!(c.run, 4); // 1 2 3 4 matched, found through its last three tokens
        c.push(5);
        assert_eq!((c.run, c.drafts(1)), (5, vec![6u32]));
    }

    #[test]
    fn choose_adds_rows_only_while_they_pay() {
        let spec = Speculation::new();
        // one stream: near-certain drafts are all taken, hopeless ones are not
        assert_eq!(spec.choose(&[(0, vec![0.99; 7])], 1, 1, 0, 4, true)[0], 7);
        assert_eq!(spec.choose(&[(0, vec![0.02; 7])], 1, 1, 0, 4, true)[0], 0);
        // a conversation's drafts are taken in order, and a weak draft blocks the ones behind it
        assert_eq!(spec.choose(&[(0, vec![0.9, 0.9, 0.01, 0.99])], 1, 1, 0, 4, true)[0], 2);
        // with more streams sharing the step, a row has to be worth more
        let alone = spec.choose(&[(0, vec![0.5; 7])], 1, 1, 0, 4, true)[0];
        let shared = spec.choose(&[(0, vec![0.5; 7])], 4, 4, 0, 4, true)[0];
        assert!(shared <= alone, "{shared} > {alone}");
        // long copied drafts fit, and never exceed the step's row limit
        let k = spec.choose(&[(0, vec![0.995; 96]), (1, vec![0.995; 96])], 2, 2, 0, 4, false);
        assert!(k[0] + k[1] + 2 <= TMAX && k[0] >= 24, "{k:?}");
        // big steps come in few sizes, small ones in any size; rows the step has anyway are never dropped
        for n in 1..=40 {
            let k = spec.choose(&[(0, vec![0.999; n]), (1, vec![0.999; n])], 2, 2, 0, 4, false);
            let rows = 2 + k[0] + k[1];
            assert!(k[0] <= n && k[1] <= n && (rows <= 16 || (rows <= 64 && rows % 4 == 0) || rows % 8 == 0), "{n}: {k:?}");
        }
        assert_eq!(spec.choose(&[(0, vec![0.999; 3])], 1 + 48, 1, 48, 4, true)[0], 3); // 49 rows anyway: 52 is a size
        assert!(spec.choose(&[(0, vec![f32::NAN; 5])], 1, 1, 0, 4, true)[0] <= 5);
        // a step that is also reading a prompt is worth more as it is, so drafts have to be better to join it
        let busy = spec.choose(&[(0, vec![0.5; 7])], 1 + 48, 1, 48, 4, true)[0];
        let idle = spec.choose(&[(0, vec![0.5; 7])], 1 + 48, 1, 0, 4, true)[0];
        assert!(busy <= idle, "{busy} > {idle}");
        // a conversation copying a long passage takes all of it when alone: 31 drafts and its next token fill 32 rows
        assert_eq!(spec.choose(&[(0, vec![0.995f32; 31])], 1, 1, 0, 4, false)[0], 31);
        // a 33rd row would start a new tile of rows in every matrix kernel: not for a doubtful draft
        assert_eq!(spec.choose(&[(0, vec![0.999; 28]), (1, vec![0.999, 0.999, 0.999, 0.3, 0.3])], 2, 2, 0, 4, false), vec![27, 3, 0, 0]);
    }

    #[test]
    fn step_time_table_is_monotonic() {
        assert_eq!(Speculation::rows_ms(1), 6.12);
        assert_eq!(Speculation::rows_ms(128), 72.32);
        assert!((Speculation::rows_ms(36) - 27.15).abs() < 0.01); // between the points at 34 and 40
        assert!((2..=128).all(|r| Speculation::rows_ms(r) > Speculation::rows_ms(r - 1)));
    }
}
