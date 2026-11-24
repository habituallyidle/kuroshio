mod chat;
mod engine;
mod safetensors;
mod server;

use anyhow::{bail, Result};
use std::path::PathBuf;

const HELP: &str = "\
kuroshio - an inference engine for Inkling 2 Mini on one RTX 5090

USAGE
  kuroshio serve [options]          OpenAI-compatible server (POST /v1/chat/completions, GET /v1/models, GET /health)
  kuroshio run \"prompt\" [options]   generate once and print the text
  kuroshio --help | --version

SERVE OPTIONS
  --host ADDR            address to listen on                               [127.0.0.1]
  --port N               port                                               [8020]
  --slots N              conversations served at once, 1-4; more wait in    [1]
                         a queue
  --ctx N|auto           longest conversation allowed, in tokens (64k);     [auto]
                         auto = what the free GPU memory could hold. Memory
                         is taken as conversations grow, and shared by slots
  --reserve SIZE         GPU memory to leave free for other programs,       [512m]
                         e.g. 6g; conversations stop growing at that point
  --idle-release SECS    give back a slot's context memory after it has     [0]
                         been idle this long; 0 = keep it (and its cache)
  --kv 16|8              key/value cache precision; 8 halves the memory     [16]
                         per token of context (about 3% slower, lossy)
  --dense 16|8           precision of the attention matrices and output     [16]
                         head; 8 is faster and saves 0.8 GiB (lossy)
  --spec K               verify up to K drafted tokens per step, 0-7;       [7]
                         0 turns speculative decoding off
  --tau auto|X           which drafts to verify: auto weighs each one's     [auto]
                         chance of acceptance against the time it costs;
                         a number 0-1 verifies those the drafter gives at
                         least that probability
  --copy N               when the conversation starts repeating an earlier  [31]
                         passage (a quoted file, rewritten code), verify up
                         to N copied tokens per step, 0-32; 0 = off
  --prefill-share N      prompt tokens processed per step while another     [48]
                         conversation is generating, 1-128
  --cors ORIGIN          let web pages from this origin call the server     [none]
                         (http://localhost:3000, or * for any)
  --sampled-drafts       when sampling (temperature >= 0.3), draw drafts from the drafter's
                         distribution instead of taking its top token (experimental: exact,
                         but measured no faster)
  --no-draft             do not load the drafter (saves 1.1 GiB; no speculation)
  --no-prefix-cache      always re-read the whole prompt (saves 0.2 GiB per slot)
  --no-tool-fix          let the model write the fixed start of a tool call itself
  --model DIR            model snapshot directory  [$INKLING_WEIGHTS, the path saved by install.sh, the Hugging Face cache]
  --name ID              model id reported to clients                       [inkling-2-mini]

RUN OPTIONS
  --effort LEVEL         none|minimal|low|medium|high|xhigh|max             [none]
  --max-tokens N                                                            [512]
  --temp X               sampling temperature; 0 = greedy                   [0]
  --seed N                                                                  [1]
  plus --spec, --tau, --kv, --dense, --no-draft, --model as above

Developer commands (see docs/): check, argmaxes, headcheck, interleave, draftcheck, samptest, nll, topk, steptime,
microbench, graphmem, gen, render, inspect.
";

/// Command-line flags: `--name value` pairs and bare `--name` switches, plus positional arguments.
struct Flags {
    named: std::collections::HashMap<String, String>,
    pos: Vec<String>,
}

impl Flags {
    const SWITCHES: [&'static str; 4] = ["no-draft", "no-prefix-cache", "no-tool-fix", "sampled-drafts"];

    fn parse(args: &[String]) -> Result<Self> {
        let (mut named, mut pos, mut i) = (std::collections::HashMap::new(), Vec::new(), 0);
        while i < args.len() {
            if let Some(name) = args[i].strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    named.insert(k.to_string(), v.to_string());
                } else if Self::SWITCHES.contains(&name) {
                    named.insert(name.to_string(), "1".into());
                } else {
                    i += 1;
                    let v = args.get(i).ok_or_else(|| anyhow::anyhow!("--{name} needs a value"))?;
                    named.insert(name.to_string(), v.clone());
                }
            } else {
                pos.push(args[i].clone());
            }
            i += 1;
        }
        Ok(Self { named, pos })
    }

    fn has(&self, k: &str) -> bool {
        self.named.contains_key(k)
    }

    /// Flag value, else the environment variable `env` (the older way to set the same thing), else the default.
    fn get<T: std::str::FromStr>(&self, k: &str, env: &str, default: T) -> Result<T> {
        match self.named.get(k).cloned().or_else(|| std::env::var(env).ok()) {
            Some(v) => v.parse().map_err(|_| anyhow::anyhow!("--{k}: cannot use the value {v:?}")),
            None => Ok(default),
        }
    }

    fn check_known(&self, known: &[&str]) -> Result<()> {
        match self.named.keys().find(|k| !known.contains(&k.as_str())) {
            Some(k) => bail!("unknown option --{k} (see kuroshio --help)"),
            None => Ok(()),
        }
    }

    /// Options shared by `serve` and `run` that the engine reads from the environment when it loads.
    fn apply_engine_options(&self) -> Result<()> {
        if let Some(dir) = self.named.get("model") {
            std::env::set_var("INKLING_WEIGHTS", dir);
        }
        for (flag, env) in [("kv", "INK_KV_FP8"), ("dense", "INK_FP8")] {
            match self.named.get(flag).map(String::as_str) {
                None | Some("16") => {}
                Some("8") => std::env::set_var(env, "1"),
                Some(v) => bail!("--{flag} must be 16 or 8, not {v:?}"),
            }
        }
        if self.has("no-draft") {
            std::env::set_var("INK_NO_DRAFT", "1");
        }
        if self.has("no-prefix-cache") {
            std::env::set_var("INK_NO_PREFIX_CACHE", "1");
        }
        if self.has("sampled-drafts") {
            std::env::set_var("INK_SAMPLED_DRAFTS", "1");
        }
        Ok(())
    }
}

/// "64k" -> 65536, "200K" -> 204800, "131072" -> 131072.
fn parse_tokens(s: &str) -> Result<usize> {
    let (digits, mult) = match s.strip_suffix(['k', 'K']) {
        Some(d) => (d, 1024),
        None => (s, 1),
    };
    Ok(digits.parse::<usize>().map_err(|_| anyhow::anyhow!("cannot read {s:?} as a number of tokens (examples: 32768, 64k, auto)"))? * mult)
}

/// "6g" -> 6144, "512m" -> 512, "800" -> 800 (MiB).
fn parse_mib(s: &str) -> Result<usize> {
    let (digits, mult) = match s.to_ascii_lowercase() {
        l if l.ends_with('g') => (l[..l.len() - 1].to_string(), 1024.0),
        l if l.ends_with('m') => (l[..l.len() - 1].to_string(), 1.0),
        l => (l, 1.0),
    };
    let v = digits.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0 && *v * mult <= 1e6).ok_or_else(|| anyhow::anyhow!("cannot read {s:?} as a memory size (examples: 512m, 6g)"))?;
    Ok((v * mult) as usize)
}

const MIB: usize = 1 << 20;
/// Recorded step graphs take up to about 260 MiB while serving; context memory never grows into that.
const GRAPH_ROOM_MIB: usize = 320;

/// GPU memory the engine takes at start-up, in bytes, before any conversation: weights, buffers and what each slot
/// needs whatever its context (ring buffers, prefix-cache snapshot). Measured on an RTX 5090.
fn vram_base(slots: usize) -> usize {
    let on = |k: &str| std::env::var(k).is_ok();
    let (kv8, dense8, draft) = (on("INK_KV_FP8"), on("INK_FP8"), !on("INK_NO_DRAFT"));
    let fixed = 20120 * MIB - if dense8 { 800 * MIB } else { 0 } - if draft { 0 } else { 1100 * MIB };
    let per_slot = if kv8 { 141 } else { 246 } * MIB + if on("INK_NO_PREFIX_CACHE") { 0 } else if kv8 { 136 } else { 231 } * MIB;
    fixed + slots * per_slot
}

/// GPU memory per token of conversation, in bytes: 8 full-attention layer visits and one drafter layer each keep a
/// key and a value of 8 x 128 numbers, plus the attention score scratch (which grows in powers of two).
fn vram_per_token() -> usize {
    let on = |k: &str| std::env::var(k).is_ok();
    (if on("INK_KV_FP8") { 16384 } else { 32768 }) + if on("INK_NO_DRAFT") { 0 } else { 4096 } + 4608
}

fn weights_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("INKLING_WEIGHTS") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME").unwrap_or_default();
    // A path saved by `./install.sh --model DIR`.
    let cfg = std::env::var("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(&home).join(".config")).join("kuroshio/model");
    if let Ok(p) = std::fs::read_to_string(&cfg) {
        if !p.trim().is_empty() {
            return Ok(PathBuf::from(p.trim()));
        }
    }
    // The Hugging Face cache: $HF_HUB_CACHE, $HF_HOME/hub, ~/.cache/huggingface/hub.
    let mut hubs: Vec<PathBuf> = Vec::new();
    hubs.extend(std::env::var("HF_HUB_CACHE").ok().map(PathBuf::from));
    hubs.extend(std::env::var("HF_HOME").ok().map(|h| PathBuf::from(h).join("hub")));
    hubs.push(PathBuf::from(&home).join(".cache/huggingface/hub"));
    for hub in hubs {
        if let Ok(snaps) = std::fs::read_dir(hub.join("models--thinkingmachines--Inkling-2-Mini/snapshots")) {
            if let Some(dir) = snaps.flatten().map(|e| e.path()).find(|p| p.join("model-00001-of-00001.safetensors").exists()) {
                return Ok(dir);
            }
        }
    }
    bail!("cannot find the model. Download it (hf download thinkingmachines/Inkling-2-Mini) or point --model / INKLING_WEIGHTS at the snapshot directory")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        // List tensors: name, dtype, shape, megabytes.
        Some("inspect") => {
            let file = args.get(2).map(String::as_str).unwrap_or("model-00001-of-00001.safetensors");
            let st = safetensors::SafeTensors::open(&weights_dir()?.join(file))?;
            let mut total = 0usize;
            for (name, t) in &st.tensors {
                let b = t.data_offsets[1] - t.data_offsets[0];
                total += b;
                println!("{:>10.2} MB  {:8} {:?}  {}", b as f64 / 1e6, t.dtype, t.shape, name);
            }
            println!("{} tensors, {:.2} GB", st.tensors.len(), total as f64 / 1e9);
        }
        // Render an OpenAI-style request body (JSON file) to the prompt text: inkling render request.json
        Some("render") => {
            let req: serde_json::Value = serde_json::from_slice(&std::fs::read(args.get(2).map(String::as_str).unwrap_or("request.json"))?)?;
            let chat = chat::Chat::load(&weights_dir()?)?;
            print!("{}", chat.render(&req["messages"], req.get("tools").filter(|v| !v.is_null()), req.get("reasoning_effort").filter(|v| !v.is_null()))?);
        }
        // Render a one-message chat and show its tokens.
        Some("prompt") => {
            let text = args.get(2).cloned().unwrap_or_else(|| "Hello!".into());
            let effort = args.get(3).map(|s| serde_json::Value::String(s.clone()));
            let chat = chat::Chat::load(&weights_dir()?)?;
            let msgs = serde_json::json!([{ "role": "user", "content": text }]);
            let rendered = chat.render(&msgs, None, effort.as_ref())?;
            let ids = chat.encode(&rendered)?;
            println!("{rendered:?}\n{} tokens: {ids:?}\nround-trip: {:?}", ids.len(), chat.decode(&ids)?);
        }
        None | Some("help") | Some("--help") | Some("-h") => print!("{HELP}"),
        Some("--version") | Some("-V") => println!("kuroshio {}", env!("CARGO_PKG_VERSION")),
        // OpenAI-compatible server. Older positional form still works: kuroshio serve [addr] [ctx] [slots]
        Some("serve") => {
            let f = Flags::parse(&args[2..])?;
            f.check_known(&["host", "port", "slots", "ctx", "reserve", "idle-release", "kv", "dense", "spec", "tau", "prefill-share", "cors", "no-draft", "no-prefix-cache", "no-tool-fix", "sampled-drafts", "copy", "model", "name"])?;
            f.apply_engine_options()?;
            let addr = match f.pos.first() {
                Some(a) => a.clone(),
                None => format!("{}:{}", f.get("host", "", "127.0.0.1".to_string())?, f.get::<u16>("port", "", 8020)?),
            };
            let slots: usize = match f.pos.get(2) {
                Some(s) => s.parse()?,
                None => f.get("slots", "", 1)?,
            };
            // Four is the design limit: the card's memory, not its speed, is what conversations compete for.
            // (INK_UNSUPPORTED_SLOTS=1 lifts it to the engine's 16, for benchmarks: the context left per conversation is tiny.)
            let most = if std::env::var("INK_UNSUPPORTED_SLOTS").is_ok() { 16 } else { 4 };
            anyhow::ensure!((1..=most).contains(&slots), "--slots must be between 1 and {most}");
            let reserve_mib = parse_mib(&f.get("reserve", "", "512m".to_string())?)? + GRAPH_ROOM_MIB;
            // Memory for conversations: what is free now, minus what the engine takes at start-up, minus the reserve.
            let free = engine::vram_free();
            let base = vram_base(slots);
            let pool = free.saturating_sub(base + reserve_mib * MIB);
            if pool < 8192 * vram_per_token() {
                bail!("not enough free GPU memory: {:.1} GiB free, {:.1} GiB needed to start with {slots} slot(s) and the reserve", free as f64 / (1u64 << 30) as f64, (base + reserve_mib * MIB + 8192 * vram_per_token()) as f64 / (1u64 << 30) as f64);
            }
            let fits = (pool / vram_per_token() / 2048 * 2048).min(1 << 20);
            let ctx_arg = f.pos.get(1).cloned().unwrap_or(f.get("ctx", "", "auto".to_string())?);
            let max_ctx = if ctx_arg == "auto" { fits } else { parse_tokens(&ctx_arg)? };
            anyhow::ensure!(max_ctx >= 2048, "--ctx must be at least 2048");
            let opts = server::Options {
                k_max: if f.has("no-draft") { 0 } else { f.get("spec", "INK_SPEC", 7usize)? },
                tau: match f.get("tau", "INK_TAU", "auto".to_string())?.as_str() {
                    "auto" => None,
                    t => Some(t.parse::<f32>().ok().filter(|t| (0.0..=1.0).contains(t)).ok_or_else(|| anyhow::anyhow!("--tau must be auto or a number between 0 and 1"))?),
                },
                prefill_share: f.get("prefill-share", "INK_PREFILL_SHARE", 48usize)?,
                prefix_cache: !f.has("no-prefix-cache") && std::env::var("INK_NO_PREFIX_CACHE").is_err(),
                tool_fix: !f.has("no-tool-fix") && std::env::var("INK_NO_TOOL_FORCE").is_err(),
                idle_release: f.get("idle-release", "", 0u64)?,
                cors: Some(f.get("cors", "", String::new())?).filter(|o| !o.is_empty()),
                copy_max: f.get("copy", "INK_COPY", 31usize)?,
            };
            anyhow::ensure!(opts.cors.as_ref().map_or(true, |o| !o.contains(['\r', '\n'])), "--cors takes an origin such as http://localhost:3000, or *");
            anyhow::ensure!(opts.k_max <= 7, "--spec must be between 0 and 7");
            // 31 drafts + the next token fill 32 rows: a run the engine can rewind, with room left to resume a conversation that ended on one.
            anyhow::ensure!(opts.copy_max + 5 <= engine::Engine::max_spec_run(), "--copy must be between 0 and {}", engine::Engine::max_spec_run() - 5);
            anyhow::ensure!((1..=128).contains(&opts.prefill_share), "--prefill-share must be between 1 and 128");
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            eprintln!(
                "{:.1} GiB of GPU memory free; the engine takes {:.1} GiB now and up to {:.1} GiB more as conversations grow (about {fits} tokens in total, shared by {slots} slot(s)), leaving {:.1} GiB",
                free as f64 / (1u64 << 30) as f64, base as f64 / (1u64 << 30) as f64, pool as f64 / (1u64 << 30) as f64, (reserve_mib * MIB) as f64 / (1u64 << 30) as f64
            );
            let mut eng = engine::Engine::load_lazy(&dir, max_ctx, slots, reserve_mib)?;
            let ctx = eng.max_ctx(); // the engine may have rounded the size up
            server::serve(&addr, &f.get("name", "", "inkling-2-mini".to_string())?, chat, &mut eng, ctx, opts)?;
        }
        // Greedy generation: inkling gen "<text>" [effort] [max_tokens]
        // INK_SPEC=k verifies up to k drafts per step (0 = no speculation), INK_TAU = drafter confidence needed to verify a draft.
        // `run`: the user-facing form (flags). `gen`: the older positional form used by the test scripts
        // (kuroshio gen [text] [effort] [max]; INK_SPEC / INK_TAU / INK_TEMP / INK_SEED).
        Some(cmd @ ("gen" | "run")) => {
            let f = Flags::parse(&args[2..])?;
            f.check_known(&["effort", "max-tokens", "temp", "seed", "spec", "tau", "kv", "dense", "no-draft", "sampled-drafts", "model"])?;
            f.apply_engine_options()?;
            let run = cmd == "run";
            let text = f.pos.first().cloned().unwrap_or_else(|| "Hello!".into());
            let effort = serde_json::Value::String(f.pos.get(1).cloned().unwrap_or(f.get("effort", "", "none".to_string())?));
            let max_tokens: usize = match f.pos.get(2) {
                Some(s) => s.parse()?,
                None => f.get("max-tokens", "", if run { 512 } else { 128 })?,
            };
            let k_max: usize = if f.has("no-draft") { 0 } else { f.get("spec", "INK_SPEC", if run { 7 } else { 0 })? };
            // (`auto` is the server's per-step choice; a single run uses the fixed threshold 0.5 for it.)
            let tau: f32 = match f.get("tau", "INK_TAU", if run { "0.5" } else { "0" }.to_string())?.as_str() {
                "auto" => 0.5,
                t => t.parse().map_err(|_| anyhow::anyhow!("--tau must be auto or a number between 0 and 1"))?,
            };
            anyhow::ensure!(k_max <= 7 && (0.0..=1.0).contains(&tau), "--spec must be 0-7 and --tau 0-1");
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let ids = chat.encode(&chat.render(&serde_json::json!([{ "role": "user", "content": text }]), None, Some(&effort))?)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            let temp: f32 = f.get("temp", "INK_TEMP", 0.0)?;
            let seed: u32 = f.get("seed", "INK_SEED", 1)?;
            eng.set_sampling(temp, seed);
            let mut rs = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1;
            let mut uniform = move || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; (rs >> 40) as f32 / (1u64 << 24) as f32 };
            let t0 = std::time::Instant::now();
            let mut next = eng.start(&ids);
            let ttft = t0.elapsed().as_secs_f64();
            let mut out = vec![next];
            eng.prof_report(ids.len());
            let t1 = std::time::Instant::now();
            let (mut stats, mut valid) = (engine::SpecStats::default(), eng.last_batch());
            'gen: while out.len() < max_tokens && next != 200006 && eng.room() >= 1 {
                for t in eng.step(next, &mut valid, k_max, tau, &mut uniform, &mut stats) {
                    out.push(t);
                    next = t;
                    if t == 200006 || out.len() >= max_tokens {
                        break 'gen;
                    }
                }
            }
            let dt = t1.elapsed().as_secs_f64();
            eng.prof_report(out.len() - 1);
            if run {
                // Print the answer; the model's thinking goes to stderr. Control tokens (>= 199,998) switch between them.
                let (mut answer, mut thinking, mut seg, mut in_think) = (String::new(), String::new(), Vec::new(), false);
                for &t in out.iter().chain([200_010u32].iter()) {
                    if t >= 199_998 {
                        if in_think { thinking.push_str(&chat.decode(&seg)?) } else { answer.push_str(&chat.decode(&seg)?) }
                        seg.clear();
                        in_think = t == 200_008 || (in_think && t != 200_010 && t != 200_004);
                    } else {
                        seg.push(t);
                    }
                }
                if !thinking.trim().is_empty() {
                    eprintln!("[thinking] {}", thinking.trim());
                }
                println!("{}", answer.trim());
            } else {
                println!("{}", chat.decode(&out)?);
            }
            eprintln!("prompt {} tokens in {:.2}s ({:.0} tok/s); {} tokens decoded at {:.1} tok/s",
                ids.len(), ttft, ids.len() as f64 / ttft, out.len() - 1, (out.len() - 1) as f64 / dt);
            if k_max > 0 {
                eprintln!("speculation: {} steps, {:.2} drafts verified/step, {:.2} accepted/step, {:.2} tokens/step, {:.2} ms/step",
                    stats.steps, stats.verified as f64 / stats.steps as f64, stats.accepted as f64 / stats.steps as f64,
                    (out.len() - 1) as f64 / stats.steps as f64, dt * 1e3 / stats.steps as f64);
            }
        }
        // Raw step timing: kuroshio steptime [context_tokens]   (batch sizes 1, 4, 8 after that much context, or those in
        // INK_STEP_T; INK_STEP_FAST=1 times steps as the server runs them: the two-stage head for up to 32 rows)
        Some("steptime") => {
            let ctx: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(64);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let ids = chat.encode(&std::fs::read_to_string(std::env::var("INK_TEXT").unwrap_or_else(|_| "docs/ARCHITECTURE.md".into()))?)?;
            let temp: f32 = std::env::var("INK_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let mut eng = engine::Engine::load(&dir, std::env::var("INK_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(8192))?;
            eng.set_sampling(temp, 1);
            // INK_STEP_T=1,2,3: the batch sizes to time
            let ts: Vec<usize> = std::env::var("INK_STEP_T").ok().map_or(vec![1, 4, 8], |v| v.split(',').filter_map(|x| x.parse().ok()).collect());
            for &t in &ts {
                eng.reset();
                eng.forward(&ids[..ctx], None);
                let fast = std::env::var("INK_STEP_FAST").is_ok();
                let run = |eng: &mut engine::Engine, off: usize| {
                    if fast {
                        let rows: Vec<(usize, usize, u32)> = (0..t).map(|i| (0, off + i, ids[off + i])).collect();
                        eng.step_rows(&rows, None, true);
                    } else {
                        eng.forward(&ids[off..off + t], None);
                    }
                };
                run(&mut eng, ctx); // capture the graph for this shape
                eng.prof_report(1);
                let (mut times, mut off) = (Vec::new(), ctx + t);
                let reps = (ids.len() - off - 16) / t;
                let reps = reps.min(300);
                for _ in 0..reps {
                    let t0 = std::time::Instant::now();
                    run(&mut eng, off);
                    times.push(t0.elapsed().as_secs_f64() * 1e3);
                    off += t;
                }
                eng.prof_report(reps);
                let series: Vec<String> = times.iter().take(48).map(|x| format!("{x:.1}")).collect();
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = times.len();
                println!("ctx {ctx:5} T={t:2}: median {:.2} ms/step ({:.0} tok/s), mean {:.2}, p90 {:.2}, max {:.2} over {n} steps", times[n / 2], t as f64 * 1e3 / times[n / 2], mean, times[n * 9 / 10], times[n - 1]);
                if t == 1 { println!("   first 48 step times: {}", series.join(" ")); }
                if t == 1 && eng.has_draft() {
                    // Time the drafter alone (7 and 4 drafts): it is replayed from its own graph.
                    for nd in [7usize, 4] {
                        let mut dt = Vec::new();
                        for _ in 0..40 {
                            let t0 = std::time::Instant::now();
                            let _ = eng.draft(ids[off], 1, nd);
                            dt.push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                        dt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        println!("   drafter alone, {nd} drafts: median {:.2} ms", dt[20]);
                    }
                }
            }
        }
        // Dump tokens + DSpark taps of a greedy generation for the Python drafter: inkling dumpfeats "<text>" N out_prefix
        Some("dumpfeats") => {
            let text = args.get(2).cloned().unwrap_or_else(|| "Hello!".into());
            let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(200);
            let prefix = args.get(4).cloned().unwrap_or_else(|| "feats".into());
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let mut toks = chat.encode(&chat.render(&serde_json::json!([{ "role": "user", "content": text }]), None, Some(&serde_json::json!("none")))?)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            let prompt_len = toks.len();
            let (mut feats, mut i) = (Vec::<f32>::new(), 0usize);
            while i < toks.len() {
                let next = eng.forward(&toks[i..i + 1], None)[0]; // one token at a time, like plain decoding
                feats.extend(eng.feats(1));
                i += 1;
                if i == toks.len() && toks.len() < n && next != 200006 {
                    toks.push(next);
                }
            }
            std::fs::write(format!("{prefix}.tokens.json"), serde_json::to_vec(&toks)?)?;
            std::fs::write(format!("{prefix}.feats.f32"), feats.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
            eprintln!("{} tokens ({} generated): {}", toks.len(), toks.len() - prompt_len, chat.decode(&toks)?.chars().take(300).collect::<String>());
        }
        // Statistical check that speculation does not change the sampling distribution: inkling samptest [trials]
        // Samples the same next token many times (temperature 1) with 0 and with 4 verified drafts and compares histograms.
        Some("samptest") => {
            let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3000);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let mut prefix = chat.encode(&chat.render(&serde_json::json!([{ "role": "user", "content": "Explain how a bill becomes law in the United States, in two paragraphs." }]), None, Some(&serde_json::json!("none")))?)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            eng.set_sampling(0.0, 0);
            let mut next = eng.start(&prefix);
            for _ in 0..14 {
                prefix.push(next);
                next = eng.forward(&[next], None)[0];
            }
            let p = prefix.len() - 1; // the anchor is prefix[p]; we sample the token at p + 1
            let mut rs = 0x1234_5678_9ABC_DEF1u64;
            let mut uniform = move || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; (rs >> 40) as f32 / (1u64 << 24) as f32 };
            let temp: f32 = std::env::var("INK_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
            let mut hists = Vec::new();
            for k in [0usize, 4] {
                let (mut h, mut stats) = (std::collections::HashMap::<u32, usize>::new(), engine::SpecStats::default());
                for i in 0..n {
                    eng.set_sampling(temp, (i as u32).wrapping_mul(2654435761).wrapping_add(k as u32));
                    eng.pos = p - 1;
                    eng.forward(&prefix[p - 1..p], None); // re-run the token before the anchor: restores the drafter's inputs
                    let mut valid = 1;
                    // INK_TAU: also exercise the choice of how many drafts to verify (it must not bias the result)
                    let out = eng.step(prefix[p], &mut valid, k, std::env::var("INK_TAU").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0), &mut uniform, &mut stats);
                    *h.entry(out[0]).or_default() += 1;
                }
                eprintln!("k={k}: {} distinct tokens; accepted drafts/step {:.2}", h.len(), stats.accepted as f64 / stats.steps as f64);
                hists.push(h);
            }
            let mut keys: Vec<u32> = hists[0].keys().chain(hists[1].keys()).cloned().collect();
            keys.sort(); keys.dedup();
            let tv: f64 = keys.iter().map(|k| (*hists[0].get(k).unwrap_or(&0) as f64 - *hists[1].get(k).unwrap_or(&0) as f64).abs()).sum::<f64>() / (2.0 * n as f64);
            keys.sort_by_key(|k| std::cmp::Reverse(hists[0].get(k).unwrap_or(&0) + hists[1].get(k).unwrap_or(&0)));
            println!("context: ...{:?}", chat.decode(&prefix[prefix.len() - 10..])?);
            for k in keys.iter().take(8) {
                println!("  {:>14?}  plain {:5.1}%   speculative {:5.1}%", chat.decode(&[*k])?, 100.0 * *hists[0].get(k).unwrap_or(&0) as f64 / n as f64, 100.0 * *hists[1].get(k).unwrap_or(&0) as f64 / n as f64);
            }
            println!("total variation distance between the two histograms: {tv:.3} ({n} samples each; two samples of the SAME distribution would give about {:.3})", (keys.len() as f64 / (std::f64::consts::PI * n as f64)).sqrt().min(1.0));
        }
        // Per-kernel timings in isolation: inkling microbench [context_tokens]
        Some("microbench") => {
            let ctx: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1500);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let ids = chat.encode(&std::fs::read_to_string("docs/ARCHITECTURE.md")?)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            eng.forward(&ids[..ctx], None);
            // INK_MB_T=1,8,16: the batch sizes to time
            let ts: Vec<usize> = std::env::var("INK_MB_T").ok().map_or(vec![1, 8], |v| v.split(',').filter_map(|x| x.parse().ok()).collect());
            for t in ts {
                eng.forward(&ids[ctx..ctx + t], None);
                eng.reset();
                eng.forward(&ids[..ctx], None);
                eng.microbench(t);
            }
        }
        // Quality metric on real text: mean negative log-likelihood (nats/token) and top-1 accuracy, teacher forced.
        // inkling nll [tokens] [file]
        Some("nll") => {
            let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4000);
            let file = args.get(3).cloned().unwrap_or_else(|| "docs/ARCHITECTURE.md".into());
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let ids = chat.encode(&std::fs::read_to_string(&file)?)?;
            let n = n.min(ids.len() - 1);
            let mut eng = engine::Engine::load(&dir, 8192)?;
            eng.set_sampling(1.0, 1); // log-probabilities are reported at temperature 1
            let targets: Vec<i32> = ids[1..=n].iter().map(|&t| t as i32).collect();
            let rows = eng.forward_full(&ids[..n], Some(&targets), None);
            let skip = 32; // the first tokens have no context; leave them out
            let nll: f64 = rows[skip..].iter().map(|r| -(r.logp_excl as f64)).sum::<f64>() / (n - skip) as f64;
            let acc = rows[skip..].iter().zip(&ids[skip + 1..=n]).filter(|(r, t)| r.greedy == **t).count() as f64 / (n - skip) as f64;
            println!("{file}: {} tokens  NLL {nll:.4} nats/token  perplexity {:.3}  top-1 accuracy {:.2}%", n - skip, nll.exp(), 100.0 * acc);
        }
        // Concurrency check: several conversations advanced together must produce exactly the tokens they produce alone.
        // inkling interleave [conversations] [tokens_each]
        // Diagnostic: GPU memory taken by the recorded step graphs (one per batch size, context bucket, with/without head).
        Some("graphmem") => {
            let dir = weights_dir()?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            let toks: Vec<u32> = (0..engine::TMAX as u32).map(|i| 1000 + i).collect();
            let f0 = engine::vram_free();
            let mut last = f0;
            // INK_GM_STEP=1: steps as the server runs them (the two-stage head for up to 32 rows) instead of `forward`
            let as_step = std::env::var("INK_GM_STEP").is_ok();
            let run = |eng: &mut engine::Engine, t: usize| {
                eng.reset();
                let t0 = std::time::Instant::now();
                if as_step {
                    let rows: Vec<(usize, usize, u32)> = (0..t).map(|i| (0, i, toks[i])).collect();
                    eng.step_rows(&rows, None, true);
                } else {
                    eng.forward(&toks[..t], None);
                }
                t0.elapsed().as_secs_f64() * 1e3
            };
            // INK_GM_ROUNDS=n: go through all sizes n more times first (every step re-records an evicted graph) and print
            // the process's own memory: a leak per recording would show as steady growth.
            let rss = || std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1)?.parse::<f64>().ok()).unwrap_or(0.0) * 4096.0 / 1048576.0;
            for round in 0..std::env::var("INK_GM_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(0usize) {
                for t in 1..=engine::TMAX {
                    run(&mut eng, t);
                }
                println!("round {round}: host memory {:.0} MiB, GPU memory used by graphs {:.0} MiB", rss(), (f0 as f64 - engine::vram_free() as f64) / 1048576.0);
            }
            for t in 1..=engine::TMAX {
                let first = run(&mut eng, t);
                let second = run(&mut eng, t);
                if t % 16 == 0 || t == 1 || t == 4 || t == 8 {
                    let f = engine::vram_free();
                    println!("T={t:3}: first call {first:6.1} ms, second {second:6.1} ms; graphs so far {:6.1} MiB", (f0 as f64 - f as f64) / 1048576.0);
                    last = f;
                }
            }
            let _ = last;
        }
        // Test of the batched drafter: inkling draftcheck [rounds]. Four conversations of different lengths advance
        // together; at every round the drafts computed in one pass must equal the drafts computed one conversation
        // at a time. Also times both ways.
        Some("draftcheck") => {
            let rounds: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(40);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let files = ["docs/ARCHITECTURE.md", "PLAN.md", "engine/engine.cu", "host/src/main.rs"];
            let texts: Vec<Vec<u32>> = files.iter().map(|f| chat.encode(&std::fs::read_to_string(f).unwrap()).unwrap()).collect();
            let mut eng = engine::Engine::load_slots(&dir, 8192, 4)?;
            anyhow::ensure!(eng.has_draft(), "the drafter is not loaded");
            let start = [40usize, 700, 1500, 2600]; // different context lengths, on both sides of the 1,024 window
            let mut pos = vec![0usize; 4];
            for s in 0..4 {
                while pos[s] < start[s] {
                    let c = 128.min(start[s] - pos[s]);
                    let rows: Vec<(usize, usize, u32)> = (0..c).map(|i| (s, pos[s] + i, texts[s][pos[s] + i])).collect();
                    eng.step_rows(&rows, None, false);
                    eng.commit(s, c - 1);
                    pos[s] += c;
                }
            }
            let (mut bad, mut t_batch, mut t_single) = (0usize, Vec::new(), Vec::new());
            for round in 0..rounds {
                // advance every conversation by 1..5 tokens in one step, then draft after the last one
                let (mut rows, mut last) = (Vec::new(), [0usize; 4]);
                for s in 0..4 {
                    let c = 1 + (round * 7 + s * 3) % 5;
                    for i in 0..c {
                        rows.push((s, pos[s] + i, texts[s][pos[s] + i]));
                    }
                    pos[s] += c;
                    last[s] = rows.len() - 1;
                }
                eng.step_rows(&rows, None, false);
                for s in 0..4 {
                    eng.commit(s, last[s]);
                }
                let reqs: Vec<(usize, u32, usize)> = (0..4).map(|s| (s, texts[s][pos[s]], pos[s])).collect();
                let n = 1 + round % 7;
                let t = std::time::Instant::now();
                let together = eng.draft_batch(&reqs, n, None).unwrap();
                t_batch.push(t.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                let single: Vec<_> = reqs.iter().map(|r| eng.draft_seq(r.0, r.1, r.2, n).unwrap()).collect();
                t_single.push(t.elapsed().as_secs_f64() * 1e3);
                for s in 0..4 {
                    let same = together[s].tokens[..n] == single[s].tokens[..n] && (0..n).all(|i| (together[s].conf[i] - single[s].conf[i]).abs() < 1e-3);
                    if !same {
                        bad += 1;
                        eprintln!("round {round} conversation {s}: together {:?} {:?} vs alone {:?} {:?}", &together[s].tokens[..n], &together[s].conf[..n], &single[s].tokens[..n], &single[s].conf[..n]);
                    }
                }
            }
            t_batch.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t_single.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("4 conversations: one pass {:.2} ms, one at a time {:.2} ms (medians over {rounds} rounds)", t_batch[rounds / 2], t_single[rounds / 2]);
            println!("{}", if bad == 0 { "drafts IDENTICAL".to_string() } else { format!("{bad} draft sets DIFFER") });
        }
        Some("interleave") => {
            let nseq: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3);
            let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1300);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let files = ["docs/ARCHITECTURE.md", "docs/IMPLEMENTATION_LOG.md", "PLAN.md", "THEORY.md", "README.md", "engine/engine.cu", "host/src/main.rs", "ref/inkling_ref.py"];
            let texts: Vec<Vec<u32>> = (0..nseq).map(|i| chat.encode(&std::fs::read_to_string(files[i % files.len()]).unwrap()).unwrap()).collect();
            let n = texts.iter().map(|t| t.len()).min().unwrap().min(n);
            // Twice as many slots as conversations: the interleaved run uses fresh slots, so nothing left over from the
            // separate runs can hide a missing write.
            let mut eng = engine::Engine::load_slots(&dir, 4096, 2 * nseq)?;
            // 1. each conversation alone, in its own slot, in chunks of 16
            let mut alone: Vec<Vec<u32>> = Vec::new();
            for (s, t) in texts.iter().enumerate() {
                let mut out = Vec::new();
                let mut p = 0;
                while p < n {
                    let c = 16.min(n - p);
                    let rows: Vec<(usize, usize, u32)> = (0..c).map(|i| (s, p + i, t[p + i])).collect();
                    out.extend(eng.step_rows(&rows, None, true).iter().map(|r| r.greedy));
                    p += c;
                }
                alone.push(out);
            }
            // 2. all conversations together, each contributing an irregular number of rows per step
            let (mut pos, mut mixed, mut steps, mut seed) = (vec![0usize; nseq], vec![Vec::new(); nseq], 0usize, 12345u64);
            // INK_IL_PRE=n: first read n tokens of every conversation as prompt-only steps of up to 256 rows that mix the
            // conversations in irregular runs (runs straddle the attention halves of a big step). What follows must still
            // match the separate runs.
            let pre = std::env::var("INK_IL_PRE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(n);
            while pos.iter().any(|&p| p < pre) {
                let mut rows = Vec::new();
                for s in 0..nseq {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let c = (1 + (seed >> 33) as usize % 200).min(pre - pos[s]).min(engine::PMAX - rows.len());
                    rows.extend((0..c).map(|i| (nseq + s, pos[s] + i, texts[s][pos[s] + i])));
                    pos[s] += c;
                }
                eng.step_rows(&rows, None, false);
                steps += 1;
            }
            for s in 0..nseq {
                mixed[s] = alone[s][..pos[s]].to_vec(); // not predicted in this run
            }
            while pos.iter().any(|&p| p < n) {
                let (mut rows, mut owner) = (Vec::new(), Vec::new());
                for s in 0..nseq {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let want = 1 + (seed >> 33) as usize % 45; // 1..45 rows: crosses the in-kernel conv history limit (37) and the kernels' 8-row groups
                    let c = want.min(n - pos[s]).min(engine::TMAX - rows.len());
                    for i in 0..c {
                        rows.push((nseq + s, pos[s] + i, texts[s][pos[s] + i]));
                        owner.push(s);
                    }
                    pos[s] += c;
                }
                if rows.is_empty() { break; }
                for (r, s) in eng.step_rows(&rows, None, true).iter().zip(&owner) {
                    mixed[*s].push(r.greedy);
                }
                steps += 1;
            }
            let mut bad = 0;
            for s in 0..nseq {
                let d = alone[s].iter().zip(&mixed[s]).filter(|(a, b)| a != b).count() + alone[s].len().abs_diff(mixed[s].len());
                println!("conversation {s}: {} tokens, {d} differ between alone and interleaved", alone[s].len());
                bad += d;
            }
            println!("{steps} mixed steps; {}", if bad == 0 { "IDENTICAL" } else { "MISMATCH" });
            if bad != 0 { bail!("interleaved conversations differ from separate runs"); }
        }
        // Greedy next-token ids over real text (teacher forced), for comparing engine variants: inkling argmaxes N
        Some("argmaxes") => {
            let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1500);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let ids = chat.encode(&std::fs::read_to_string(std::env::var("INK_TEXT").unwrap_or_else(|_| "docs/ARCHITECTURE.md".into()))?)?;
            let mut eng = engine::Engine::load(&dir, std::env::var("INK_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(8192))?;
            let chunk: usize = std::env::var("INK_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
            let mut out = Vec::new();
            // INK_PRE=tokens,rows: read that many tokens first as prompt-only steps of `rows` rows (no output head, up to
            // 256 rows), then continue as usual. Their predictions are printed as 0.
            let pre: Vec<usize> = std::env::var("INK_PRE").ok().map_or(vec![], |v| v.split(',').filter_map(|x| x.parse().ok()).collect());
            let skip = if pre.len() == 2 { pre[0].min(n) } else { 0 };
            for (i, c) in ids[..skip].chunks(pre.get(1).copied().unwrap_or(1).clamp(1, engine::PMAX)).enumerate() {
                let p0 = i * pre[1].clamp(1, engine::PMAX);
                let rows: Vec<(usize, usize, u32)> = c.iter().enumerate().map(|(j, &t)| (0, p0 + j, t)).collect();
                let t0 = std::time::Instant::now();
                eng.step_rows(&rows, None, false);
                eprintln!("prompt step of {} rows at {p0}: {:.2} ms", c.len(), t0.elapsed().as_secs_f64() * 1e3);
                out.extend(std::iter::repeat(0u32).take(c.len()));
            }
            if skip > 0 && std::env::var("INK_PROF").is_ok() {
                eng.prof_report((skip / pre[1].clamp(1, engine::PMAX)).max(1)); // per prompt step
            }
            eng.set_pos(skip);
            for c in ids[skip..n].chunks(chunk) {
                out.extend(eng.forward(c, None));
            }
            let hits = out.iter().zip(&ids[1..=n]).filter(|(a, b)| a == b).count();
            eprintln!("next-token accuracy on the text: {:.1}%", 100.0 * hits as f64 / n as f64);
            println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        }
        // The two-stage output head against the full one: kuroshio headcheck [tokens]. Steps of 1..32 rows at temperatures
        // from 0 to 2 over three texts; after each, both heads run again on the same rows. A different argmax or sample
        // (from the step itself or from the re-run), or a token whose approximation error exceeds its bound, is a failure.
        Some("headcheck") => {
            let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3000);
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            let (mut steps, mut rows_n, mut diff, mut worst, mut ratio, mut blocks, mut most, mut over) = (0usize, 0usize, 0.0f64, f64::MIN, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for file in ["docs/ARCHITECTURE.md", "host/src/server.rs", "LICENSE"] {
                let ids = chat.encode(&std::fs::read_to_string(file)?)?;
                let n = n.min(ids.len());
                eng.reset();
                let (mut p, mut k) = (0usize, 0usize);
                while p < n {
                    let c = [1usize, 2, 3, 5, 8, 13, 16, 24, 32, 7][k % 10].min(n - p);
                    // every third step one temperature for all rows, otherwise greedy and sampling rows mixed
                    let temps = [0.0f32, 0.6, 1.0, 0.05, 0.3, 2.0];
                    let rows: Vec<(usize, usize, u32)> = (0..c).map(|i| (0, p + i, ids[p + i])).collect();
                    let sampling: Vec<(f32, u32)> = (0..c).map(|i| (temps[if k % 3 == 0 { (k / 3) % 3 } else { (k + i) % 6 }], 1000 + (k * 37 + i) as u32)).collect();
                    eng.step_rows_with(&rows, None, true, Some(&sampling), None);
                    let r = eng.headcheck().ok_or_else(|| anyhow::anyhow!("the two-stage head is not available in this configuration"))?;
                    (steps, rows_n, diff, worst, ratio, blocks, most, over) = (steps + 1, rows_n + c, diff + r[0], worst.max(r[1]), ratio.max(r[2]), blocks + r[3], most.max(r[3]), over + r[4]);
                    p += c;
                    k += 1;
                }
            }
            println!("{steps} steps, {rows_n} rows: {diff} rows differ from the full head; worst error - bound {worst:.3} (must be negative), worst error / bound {ratio:.3};");
            println!("candidate blocks per row {:.1} on average, {most} at most in a step, {over} steps over the limit (those use the full head)", blocks / rows_n as f64);
            anyhow::ensure!(diff == 0.0 && worst < 0.0, "the two-stage head disagrees with the full head");
            println!("IDENTICAL");
        }
        // Teacher-forced comparison with golden outputs: inkling check golden.json [trace_out.bin]
        // Diagnostic: inkling topk tokens.json [last_n] - run the tokens, print the model's 5 most likely next tokens
        // (with probabilities) after each of the last n positions.
        Some("topk") => {
            let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(args.get(2).map(String::as_str).unwrap_or("tokens.json"))?)?;
            let n: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16).min(ids.len());
            let dir = weights_dir()?;
            let chat = chat::Chat::load(&dir)?;
            let mut eng = engine::Engine::load(&dir, 8192)?;
            let first = ids.len() - n;
            for chunk in ids[..first].chunks(engine::TMAX) {
                eng.forward(chunk, None);
            }
            let mut logits = Vec::new();
            for i in first..ids.len() {
                eng.forward(&ids[i..i + 1], Some(&mut logits));
                let mx = logits.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let z: f64 = logits.iter().map(|&l| ((l as f64) - mx).exp()).sum();
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
                let top: Vec<String> = idx[..5].iter().map(|&t| format!("{:?} {:.3}", chat.decode(&[t as u32]).unwrap_or_default(), ((logits[t] as f64) - mx).exp() / z)).collect();
                let next = ids.get(i + 1).map(|&t| format!("{:?}", chat.decode(&[t]).unwrap_or_default())).unwrap_or_default();
                println!("after {:>14} -> {}   [sequence continues with {next}]", format!("{:?}", chat.decode(&ids[i..i + 1])?), top.join(" | "));
            }
        }
        Some("check") => {
            let golden: serde_json::Value = serde_json::from_slice(&std::fs::read(args.get(2).map(String::as_str).unwrap_or("tests/oracle/golden.json"))?)?;
            let mut eng = engine::Engine::load(&weights_dir()?, 8192)?;
            let mut trace = vec![0f32; 48 * 4 * 2304];
            for case in golden["cases"].as_array().unwrap() {
                eng.reset();
                let prompt: Vec<u32> = case["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                if let Some(path) = args.get(3) {
                    // Dump the per-visit activations of the first prompt token (position 0) for layer-by-layer comparison.
                    // INK_TRACE_POS=n: trace the token at position n (after running the n tokens before it) instead of token 0.
                    let tp: usize = std::env::var("INK_TRACE_POS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
                    let mut all = prompt.clone();
                    all.extend(case["steps"].as_array().unwrap().iter().map(|v| v["token"].as_u64().unwrap() as u32));
                    if tp > 0 {
                        eng.forward(&all[..tp], None);
                    }
                    eng.set_trace(&mut trace);
                    eng.forward(&all[tp..tp + 1], None);
                    let bytes: Vec<u8> = trace.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(path, bytes)?;
                    eprintln!("wrote trace of token {tp} of case {} to {path}", case["name"]);
                    return Ok(());
                }
                eng.forward(&prompt[..prompt.len() - 1], None);
                let (mut cur, mut mism, mut worst, mut n) = (*prompt.last().unwrap(), 0, 0f64, 0);
                let mut logits = Vec::new();
                for step in case["steps"].as_array().unwrap() {
                    let arg = eng.forward(&[cur], Some(&mut logits))[0];
                    let mx = logits.iter().cloned().fold(f32::MIN, f32::max) as f64;
                    let lse = mx + logits.iter().map(|&l| ((l as f64) - mx).exp()).sum::<f64>().ln();
                    for pair in step["top"].as_array().unwrap() {
                        let (id, lp) = (pair[0].as_u64().unwrap() as usize, pair[1].as_f64().unwrap());
                        if lp > -12.0 {
                            worst = worst.max(((logits[id] as f64 - lse) - lp).abs());
                        }
                    }
                    let tok = step["token"].as_u64().unwrap() as u32;
                    mism += (arg != tok) as usize;
                    n += 1;
                    cur = tok;
                }
                println!("{:6} {n} steps: argmax mismatches {mism}, max |dlogprob| (top, > -12) = {worst:.3}", case["name"].as_str().unwrap());
            }
        }
        Some(other) => bail!("unknown command {other:?} (see kuroshio --help)"),
    }
    Ok(())
}
