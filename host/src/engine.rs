//! Safe wrapper around the CUDA engine (engine/engine.cu).
use crate::safetensors::SafeTensors;
use anyhow::Result;
use std::{ffi::{c_char, c_int, c_void, CString}, path::Path, time::Instant};

extern "C" {
    fn ink_new(max_ctx: c_int, nseq: c_int, reserve_mib: c_int) -> *mut c_void;
    fn ink_ensure(e: *mut c_void, seq: c_int, tokens: c_int) -> c_int;
    fn ink_ctx_mapped(e: *mut c_void, seq: c_int) -> c_int;
    fn ink_trim(e: *mut c_void, seq: c_int, tokens: c_int);
    fn ink_step(e: *mut c_void, t: c_int, tokens: *const c_int, seqs: *const c_int, pos: *const c_int, excl: *const c_int, res: *mut c_int, logits: *mut f32, want_head: c_int, temps: *const f32, seeds: *const u32, qrows: *const c_int) -> c_int;
    fn ink_commit(e: *mut c_void, seq: c_int, row: c_int);
    fn ink_draft_seq(e: *mut c_void, seq: c_int, anchor: c_int, l: c_int, ndraft: c_int, tokens: *mut c_int, conf: *mut f32, temp: f32, seed: u32, top: *mut f32) -> c_int;
    fn ink_nseq(e: *mut c_void) -> c_int;
    fn ink_max_rewind_run() -> c_int;
    fn ink_headcheck(e: *mut c_void, out: *mut f64) -> c_int;
    fn ink_vram_free() -> usize;
    fn ink_draft_batch(e: *mut c_void, n: c_int, seqs: *const c_int, anchors: *const c_int, ls: *const c_int, ndraft: c_int, tokens: *mut c_int, conf: *mut f32, temps: *const f32, seeds: *const u32, top: *mut f32) -> c_int;
    fn ink_snapshot(e: *mut c_void, seq: c_int) -> c_int;
    fn ink_restore(e: *mut c_void, seq: c_int) -> c_int;
    fn ink_upload(e: *mut c_void, name: *const c_char, data: *const c_void, bytes: usize);
    fn ink_finalize(e: *mut c_void);
    fn ink_set_trace(e: *mut c_void, buf: *mut f32);
    fn ink_prof_report(tokens: c_int);
    fn ink_has_draft(e: *mut c_void) -> c_int;
    fn ink_draft(e: *mut c_void, anchor: c_int, l: c_int, fidx: c_int, ndraft: c_int, tokens: *mut c_int, conf: *mut f32, temp: f32, seed: u32, top: *mut f32) -> c_int;
    fn ink_max_ctx(e: *mut c_void) -> c_int;
    fn ink_microbench(e: *mut c_void, t: c_int);
    fn ink_get_feats(e: *mut c_void, t: c_int, out: *mut f32);
    fn ink_forward(e: *mut c_void, tokens: *const c_int, t: c_int, pos0: c_int, excl: *const c_int, res: *mut c_int, logits: *mut f32, want_head: c_int, qrows: *const c_int);
    fn ink_set_sampling(e: *mut c_void, temperature: f32, seed: u32);
}

pub const VOCAB: usize = 200_058;
/// Most rows of a step that uses the output head (the engine calls this HMAX; its own TMAX is `PMAX` here).
pub const TMAX: usize = 128;
/// Most rows of a step that does not use the output head (prompt rows only).
pub const PMAX: usize = 1024;
/// Whether drafts for a conversation at this temperature are sampled from the drafter's distribution (and accepted
/// with probability min(1, p/q)) instead of being its most likely tokens. Off unless asked for (`--sampled-drafts`,
/// INK_SAMPLED_DRAFTS=1): on Inkling 2 Mini it was measured to be no faster than most-likely-token drafts once the
/// number of drafts to verify is chosen by expected gain (238 vs 241 tok/s at temperature 0.6, 232 vs 236 at 1.0).
pub fn sampled_drafts(temperature: f32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    temperature >= 0.3 && *ON.get_or_init(|| std::env::var("INK_SAMPLED_DRAFTS").is_ok())
}

/// Conversations the drafter handles in one pass (DBATCH in engine.cu).
pub const DRAFT_BATCH: usize = 4;

#[derive(Default, Debug)]
pub struct SpecStats {
    pub steps: usize,
    pub verified: usize,
    pub accepted: usize,
}

/// Drafts for one conversation: the tokens, the probability the drafter gives each of them, and the largest
/// probability in its distribution at each position. The last one says how sure the drafter is without depending on
/// the token drawn; when drafts are sampled it is the only one that may decide how many to verify.
#[derive(Clone, Copy)]
pub struct Drafts {
    pub tokens: [u32; 7],
    pub conf: [f32; 7],
    pub top: [f32; 7],
}

/// What the GPU sampler returns for one position.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub greedy: u32,
    pub sample: u32,
    pub sample_excl: u32,
    pub logp_excl: f32,
}

pub struct Engine {
    raw: *mut c_void,
    pub pos: usize,
    last_batch: usize, // size of the last batch sent to the GPU (the drafter indexes into it)
    temperature: f32,
    seed: u32,
    last_only: bool, // set while processing a prompt: only the final position's prediction is wanted
}

/// Free GPU memory right now, in bytes.
pub fn vram_free() -> usize {
    unsafe { ink_vram_free() }
}

impl Engine {
    pub fn load(model_dir: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_slots(model_dir, max_ctx, 1)
    }

    /// Loads the model with memory for `max_ctx` tokens attached for every slot up front.
    pub fn load_slots(model_dir: &Path, max_ctx: usize, slots: usize) -> Result<Self> {
        let mut e = Self::load_lazy(model_dir, max_ctx, slots, 0)?;
        for s in 0..e.slots() {
            anyhow::ensure!(e.ensure(s, e.max_ctx()).is_some(), "not enough GPU memory for {slots} slot(s) of {max_ctx} tokens");
        }
        Ok(e)
    }

    /// Loads the model. A conversation may grow to `max_ctx` tokens; the memory for it is attached as it grows
    /// (`ensure`), leaving at least `reserve_mib` MiB of the GPU free.
    pub fn load_lazy(model_dir: &Path, max_ctx: usize, slots: usize, reserve_mib: usize) -> Result<Self> {
        let t0 = Instant::now();
        let st = SafeTensors::open(&model_dir.join("model-00001-of-00001.safetensors"))?;
        let raw = unsafe { ink_new(max_ctx as c_int, slots as c_int, reserve_mib as c_int) };
        let mut bytes = 0usize;
        for name in st.tensors.keys() {
            if name.starts_with("model.visual.") {
                continue; // vision tower: not needed for text
            }
            let data = st.bytes(name)?;
            bytes += data.len();
            let cname = CString::new(name.as_str())?;
            unsafe { ink_upload(raw, cname.as_ptr(), data.as_ptr() as *const c_void, data.len()) };
        }
        // The DSpark drafter ships as a separate file; its tensors get an "mtp." prefix.
        let mtp_path = model_dir.join("mtp.safetensors");
        if mtp_path.exists() && std::env::var("INK_NO_DRAFT").is_err() {
            let mtp = SafeTensors::open(&mtp_path)?;
            for name in mtp.tensors.keys() {
                let data = mtp.bytes(name)?;
                bytes += data.len();
                let cname = CString::new(format!("mtp.{name}"))?;
                unsafe { ink_upload(raw, cname.as_ptr(), data.as_ptr() as *const c_void, data.len()) };
            }
        }
        unsafe { ink_finalize(raw) };
        eprintln!("loaded {:.1} GB of weights in {:.1}s", bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
        Ok(Self { raw, pos: 0, last_batch: 0, temperature: 0.0, seed: 1, last_only: false })
    }

    /// Feed tokens at the current position; returns the greedy next token after each one.
    pub fn forward(&mut self, tokens: &[u32], logits: Option<&mut Vec<f32>>) -> Vec<u32> {
        self.forward_full(tokens, None, logits).iter().map(|r| r.greedy).collect()
    }

    /// Like `forward`, with everything the sampler produced per position. `excl[i]` (optional) is the draft
    /// token proposed for the position that row `i` predicts.
    pub fn forward_full(&mut self, tokens: &[u32], excl: Option<&[i32]>, logits: Option<&mut Vec<f32>>) -> Vec<Row> {
        self.forward_q(tokens, excl, logits, None)
    }

    /// As `forward_full`; `qrows` marks the rows that verify a sampled draft (see `step`).
    fn forward_q(&mut self, tokens: &[u32], excl: Option<&[i32]>, mut logits: Option<&mut Vec<f32>>, qrows: Option<&[i32]>) -> Vec<Row> {
        let mut out = Vec::with_capacity(tokens.len());
        if let Some(l) = logits.as_deref_mut() {
            l.clear();
        }
        let mut done = 0;
        for chunk in tokens.chunks(TMAX) {
            let toks: Vec<c_int> = chunk.iter().map(|&t| t as c_int).collect();
            let mut res = vec![0 as c_int; chunk.len() * 4];
            let mut lg = vec![0f32; if logits.is_some() { chunk.len() * VOCAB } else { 0 }];
            let lp = if logits.is_some() { lg.as_mut_ptr() } else { std::ptr::null_mut() };
            let ex = excl.map(|e| e[done..done + chunk.len()].as_ptr()).unwrap_or(std::ptr::null());
            // Only the last chunk's predictions are needed when processing a prompt; earlier chunks skip the output head.
            let head = !self.last_only || done + chunk.len() == tokens.len();
            let qr = qrows.map(|q| q[done..done + chunk.len()].as_ptr()).unwrap_or(std::ptr::null());
            unsafe { ink_forward(self.raw, toks.as_ptr(), chunk.len() as c_int, self.pos as c_int, ex, res.as_mut_ptr(), lp, head as c_int, qr) };
            self.pos += chunk.len();
            self.last_batch = chunk.len();
            done += chunk.len();
            out.extend(res.chunks(4).map(|r| Row { greedy: r[0] as u32, sample: r[1] as u32, sample_excl: r[2] as u32, logp_excl: f32::from_bits(r[3] as u32) }));
            if let Some(l) = logits.as_deref_mut() {
                l.extend_from_slice(&lg);
            }
        }
        out
    }

    /// One step over rows that may belong to different conversations: (conversation, position, token). Rows of one
    /// conversation must be adjacent and in order. Returns each row's `greedy` and `sample`. `sample_excl` and
    /// `logp_excl` are only meaningful for a step with a row that verifies a sampled draft (`step_rows_with`'s `qrows`):
    /// other steps may use the two-stage output head, which does not compute them.
    pub fn step_rows(&mut self, rows: &[(usize, usize, u32)], excl: Option<&[i32]>, head: bool) -> Vec<Row> {
        self.step_rows_with(rows, excl, head, None, None)
    }

    /// As `step_rows`, with a (temperature, seed) per row.
    /// `qrows`: for a row that verifies a sampled draft, conversation * 7 + draft index; -1 otherwise.
    pub fn step_rows_with(&mut self, rows: &[(usize, usize, u32)], excl: Option<&[i32]>, head: bool, sampling: Option<&[(f32, u32)]>, qrows: Option<&[i32]>) -> Vec<Row> {
        assert!(!rows.is_empty() && rows.len() <= if head { TMAX } else { PMAX });
        let toks: Vec<c_int> = rows.iter().map(|r| r.2 as c_int).collect();
        let seqs: Vec<c_int> = rows.iter().map(|r| r.0 as c_int).collect();
        let pos: Vec<c_int> = rows.iter().map(|r| r.1 as c_int).collect();
        let mut res = vec![0 as c_int; rows.len() * 4];
        let ex = excl.map(|e| e.as_ptr()).unwrap_or(std::ptr::null());
        let (temps, seeds): (Vec<f32>, Vec<u32>) = sampling.map(|s| s.iter().cloned().unzip()).unwrap_or_default();
        let (tp, sp) = if sampling.is_some() { (temps.as_ptr(), seeds.as_ptr()) } else { (std::ptr::null(), std::ptr::null()) };
        assert!(excl.map_or(true, |e| e.len() >= rows.len()) && sampling.map_or(true, |s| s.len() >= rows.len()) && qrows.map_or(true, |q| q.len() >= rows.len()));
        let qr = qrows.map(|q| q.as_ptr()).unwrap_or(std::ptr::null());
        // Callers of this function read each row's argmax and sample only - unless a row verifies a sampled draft,
        // which needs probabilities. Without such rows the engine may use its two-stage output head (2).
        let want = if !head { 0 } else if qrows.map_or(true, |q| q[..rows.len()].iter().all(|&x| x < 0)) { 2 } else { 1 };
        let rc = unsafe { ink_step(self.raw, rows.len() as c_int, toks.as_ptr(), seqs.as_ptr(), pos.as_ptr(), ex, res.as_mut_ptr(), std::ptr::null_mut(), want as c_int, tp, sp, qr) };
        assert_eq!(rc, 0, "the engine rejected a step: rows must be grouped by conversation, at consecutive positions, in range");
        self.last_batch = rows.len();
        res.chunks(4).map(|r| Row { greedy: r[0] as u32, sample: r[1] as u32, sample_excl: r[2] as u32, logp_excl: f32::from_bits(r[3] as u32) }).collect()
    }

    /// Re-run the output head for the last step's rows both ways and compare (developer check). Returns
    /// [rows that differ, worst (error - bound), worst error / bound, candidate blocks, overflowed].
    pub fn headcheck(&mut self) -> Option<[f64; 5]> {
        let mut out = [0f64; 5];
        (unsafe { ink_headcheck(self.raw, out.as_mut_ptr()) } == 0).then_some(out)
    }

    /// Position at which `forward` continues (after rows were fed with `step_rows`).
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Tell the drafter which row of the last step is conversation `seq`'s last valid position.
    pub fn commit(&mut self, seq: usize, row: usize) {
        unsafe { ink_commit(self.raw, seq as c_int, row as c_int) };
    }

    /// Drafts for conversation `seq`: `anchor` will sit at position `l`. None if drafting is not possible there.
    pub fn draft_seq(&mut self, seq: usize, anchor: u32, l: usize, n: usize) -> Option<Drafts> {
        let (mut t, mut c, mut p) = ([0 as c_int; 7], [0f32; 7], [0f32; 7]);
        let rc = unsafe { ink_draft_seq(self.raw, seq as c_int, anchor as c_int, l as c_int, n.clamp(1, 7) as c_int, t.as_mut_ptr(), c.as_mut_ptr(), 0.0, 0, p.as_mut_ptr()) };
        (rc == 0).then(|| Drafts { tokens: t.map(|x| x as u32), conf: c, top: p })
    }

    /// Save conversation `seq`'s ring buffers (sliding-window caches, convolution history, drafter window) so the
    /// conversation can later be resumed at the current position. False if snapshots are disabled.
    pub fn snapshot(&mut self, seq: usize) -> bool {
        unsafe { ink_snapshot(self.raw, seq as c_int) == 0 }
    }

    /// Put conversation `seq`'s ring buffers back to the last snapshot.
    pub fn restore(&mut self, seq: usize) -> bool {
        unsafe { ink_restore(self.raw, seq as c_int) == 0 }
    }

    /// Drafts for several conversations in one pass: `reqs` = (slot, anchor token, anchor position), at most
    /// `DRAFT_BATCH` of them. None if drafting is not possible for one of them.
    /// `sampling`: per conversation (temperature, seed); a temperature of 0.3 or more makes its drafts samples.
    pub fn draft_batch(&mut self, reqs: &[(usize, u32, usize)], n: usize, sampling: Option<&[(f32, u32)]>) -> Option<Vec<Drafts>> {
        if reqs.is_empty() || reqs.len() > DRAFT_BATCH {
            return None;
        }
        let seqs: Vec<c_int> = reqs.iter().map(|r| r.0 as c_int).collect();
        let anchors: Vec<c_int> = reqs.iter().map(|r| r.1 as c_int).collect();
        let ls: Vec<c_int> = reqs.iter().map(|r| r.2 as c_int).collect();
        let (mut t, mut c, mut p) = (vec![0 as c_int; reqs.len() * 7], vec![0f32; reqs.len() * 7], vec![0f32; reqs.len() * 7]);
        let (temps, seeds): (Vec<f32>, Vec<u32>) = sampling.map(|s| s.iter().cloned().unzip()).unwrap_or_default();
        let (tp, sp) = if sampling.map_or(false, |s| s.len() >= reqs.len()) { (temps.as_ptr(), seeds.as_ptr()) } else { (std::ptr::null(), std::ptr::null()) };
        let rc = unsafe { ink_draft_batch(self.raw, reqs.len() as c_int, seqs.as_ptr(), anchors.as_ptr(), ls.as_ptr(), n.clamp(1, 7) as c_int, t.as_mut_ptr(), c.as_mut_ptr(), tp, sp, p.as_mut_ptr()) };
        (rc == 0).then(|| (0..reqs.len()).map(|b| Drafts { tokens: std::array::from_fn(|i| t[b * 7 + i] as u32), conf: std::array::from_fn(|i| c[b * 7 + i]), top: std::array::from_fn(|i| p[b * 7 + i]) }).collect())
    }

    /// Make sure conversation `seq` has memory for positions below `tokens`. Returns how many it now has memory for,
    /// or None if the GPU cannot give more (what it has stays usable).
    pub fn ensure(&mut self, seq: usize, tokens: usize) -> Option<usize> {
        let rc = unsafe { ink_ensure(self.raw, seq as c_int, tokens.min(i32::MAX as usize) as c_int) };
        (rc == 0).then(|| self.ctx_mapped(seq))
    }

    /// Tokens of context conversation `seq` has memory for right now.
    pub fn ctx_mapped(&self, seq: usize) -> usize {
        unsafe { ink_ctx_mapped(self.raw, seq as c_int) as usize }
    }

    /// Give back conversation `seq`'s memory above `tokens` positions.
    pub fn trim(&mut self, seq: usize, tokens: usize) {
        unsafe { ink_trim(self.raw, seq as c_int, tokens as c_int) };
    }

    /// Longest run of rows of one conversation in a step (its next token plus drafts) that may be partly rejected:
    /// a longer run does not keep the conv history its own beginning would need to restart from.
    pub fn max_spec_run() -> usize {
        unsafe { ink_max_rewind_run() as usize }
    }

    pub fn slots(&self) -> usize {
        unsafe { ink_nseq(self.raw) as usize }
    }

    /// Temperature and seed for the sampler (temperature <= 0: greedy).
    pub fn set_sampling(&mut self, temperature: f32, seed: u32) {
        (self.temperature, self.seed) = (temperature, seed);
        unsafe { ink_set_sampling(self.raw, temperature, seed) };
    }

    /// Process a prompt and return the first generated token (sampled or greedy per `set_sampling`).
    pub fn start(&mut self, prompt: &[u32]) -> u32 {
        self.last_only = true;
        let rows = self.forward_full(prompt, None, None);
        self.last_only = false;
        let r = *rows.last().unwrap();
        if self.temperature > 0.0 { r.sample } else { r.greedy }
    }

    /// One decoding step from `anchor` (the last emitted token, not yet run through the model).
    /// With k_max > 0 the drafter proposes tokens and up to k_max of them (those it is at least `tau`
    /// confident about) are verified in the same forward pass. Returns the newly determined tokens.
    /// Greedy: a draft is kept while it equals the model's own choice. Sampling: a draft d is kept with
    /// probability p(d); on the first rejection the token is drawn from p with d removed. Both rules leave
    /// the output distribution exactly as without speculation.
    pub fn step(&mut self, anchor: u32, valid: &mut usize, k_max: usize, tau: f32, uniform: &mut dyn FnMut() -> f32, stats: &mut SpecStats) -> Vec<u32> {
        let (mut drafts, mut confs, mut k) = ([0u32; 7], [0f32; 7], 0);
        let sampled = sampled_drafts(self.temperature);
        assert!(self.room() >= 1, "context is full: check Engine::room() before calling step()");
        // Near the end of the context there is no room for a speculative batch: fall back to plain decoding.
        let k_cap = k_max.min(7).min(self.room().saturating_sub(8));
        if k_cap > 0 && self.has_draft() {
            if let Some(d) = self.draft(anchor, *valid, k_cap) {
                (drafts, confs) = (d.tokens, d.conf);
                while k < k_cap && d.top[k] >= tau {
                    k += 1;
                }
            }
        }
        let l = self.pos;
        let mut toks = vec![anchor];
        toks.extend_from_slice(&drafts[..k]);
        let mut excl: Vec<i32> = drafts[..k].iter().map(|&d| d as i32).collect();
        excl.push(-1);
        // Row i's prediction is checked against draft i. A sampled draft also needs the drafter's distribution there.
        let qrows: Vec<i32> = (0..=k).map(|i| if sampled && i < k { i as i32 } else { -1 }).collect();
        let rows = self.forward_q(&toks, Some(&excl), None, Some(&qrows));
        let (mut n, mut last) = (0, None);
        if self.temperature > 0.0 && sampled {
            // Drafts drawn from the drafter's distribution q: kept with probability min(1, p / q).
            while n < k {
                if uniform() < (rows[n].logp_excl.exp() / confs[n].max(1e-30)).min(1.0) {
                    n += 1;
                } else {
                    last = Some(rows[n].sample_excl);
                    break;
                }
            }
        } else if self.temperature > 0.0 {
            // A draft is kept exactly when it is the token the sampler draws at that position anyway (probability
            // p(draft)), so the output depends only on the seed, not on how much was speculated.
            while n < k && rows[n].sample == drafts[n] {
                n += 1;
            }
        } else {
            while n < k && rows[n].greedy == drafts[n] {
                n += 1;
            }
        }
        let last = last.unwrap_or(if self.temperature > 0.0 { rows[n].sample } else { rows[n].greedy });
        // Positions after l + n hold rejected tokens; rewinding is enough because every cache is indexed by
        // position and is overwritten when those positions are processed again.
        self.pos = l + n + 1;
        *valid = n + 1;
        stats.steps += 1;
        stats.verified += k;
        stats.accepted += n;
        let mut out = drafts[..n].to_vec();
        out.push(last);
        out
    }

    /// Record per-visit activations of the last token of each forward call into `buf` ([48][4][2304]).
    pub fn set_trace(&mut self, buf: &mut [f32]) {
        assert!(buf.len() >= 48 * 4 * 2304);
        unsafe { ink_set_trace(self.raw, buf.as_mut_ptr()) };
    }

    pub fn has_draft(&self) -> bool {
        unsafe { ink_has_draft(self.raw) != 0 }
    }

    /// Ask the drafter for 7 tokens following `anchor`, where `anchor` will sit at position `self.pos`.
    /// `valid` = how many positions of the last forward batch are still valid (after a rejection, fewer than all).
    /// Only the first `n` drafts are computed (the recurrent head runs once per draft).
    pub fn draft(&mut self, anchor: u32, valid: usize, n: usize) -> Option<Drafts> {
        let (mut t, mut c, mut p) = ([0 as c_int; 7], [0f32; 7], [0f32; 7]);
        let temp = if sampled_drafts(self.temperature) { self.temperature } else { 0.0 };
        let rc = unsafe { ink_draft(self.raw, anchor as c_int, self.pos as c_int, valid as c_int - 1, n.clamp(1, 7) as c_int, t.as_mut_ptr(), c.as_mut_ptr(), temp, self.seed, p.as_mut_ptr()) };
        (rc == 0).then(|| Drafts { tokens: t.map(|x| x as u32), conf: c, top: p })
    }

    /// Context length the engine was created with.
    pub fn max_ctx(&self) -> usize {
        unsafe { ink_max_ctx(self.raw) as usize }
    }

    /// Positions still free. A decoding step needs at least 1; speculation uses up to 8.
    pub fn room(&self) -> usize {
        self.max_ctx().saturating_sub(self.pos)
    }

    pub fn last_batch(&self) -> usize {
        self.last_batch
    }

    /// DSpark taps of the last forward call: [t][4][2304].
    pub fn feats(&self, t: usize) -> Vec<f32> {
        let mut v = vec![0f32; t * 4 * 2304];
        unsafe { ink_get_feats(self.raw, t as c_int, v.as_mut_ptr()) };
        v
    }

    /// Time each kernel type in isolation (prints a table).
    pub fn microbench(&self, t: usize) {
        unsafe { ink_microbench(self.raw, t as c_int) };
    }

    pub fn prof_report(&self, tokens: usize) {
        unsafe { ink_prof_report(tokens as c_int) };
    }

    pub fn reset(&mut self) {
        self.pos = 0; // caches are position-indexed, so old entries are simply overwritten
    }
}
