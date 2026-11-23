//! Tokenizer + chat template: turns OpenAI-style messages into the token ids the model was trained on.
use anyhow::{anyhow, Result};
use minijinja::value::{Kwargs, Value};
use minijinja::{context, Environment, Error, ErrorKind};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Chat {
    pub tokenizer: Tokenizer,
    template: String,
}

impl Chat {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        // `{% generation %}` is a Hugging Face extension that only marks assistant spans for
        // training masks; it does not change the rendered text, and minijinja doesn't know it.
        let mut template = std::fs::read_to_string(model_dir.join("chat_template.jinja"))?;
        for tag in ["{%- generation -%}", "{%- endgeneration -%}", "{% generation %}", "{% endgeneration %}"] {
            template = template.replace(tag, "");
        }
        Ok(Self { tokenizer, template })
    }

    /// OpenAI clients send tool-call arguments as a JSON *string*; the model's template wants a parsed object.
    /// The "developer" role some clients use for instructions is the template's "system".
    pub fn canonicalize(messages: &serde_json::Value) -> serde_json::Value {
        let mut m = messages.clone();
        for msg in m.as_array_mut().into_iter().flatten() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("developer") {
                msg["role"] = "system".into();
            }
            for call in msg.get_mut("tool_calls").and_then(|c| c.as_array_mut()).into_iter().flatten() {
                if let Some(args) = call.pointer_mut("/function/arguments") {
                    if let Some(parsed) = args.as_str().and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok()) {
                        *args = parsed;
                    }
                }
            }
        }
        m
    }

    /// Render messages to the prompt string. `reasoning_effort` is a name ("low") or a number in [0, 0.99].
    pub fn render(&self, messages: &serde_json::Value, tools: Option<&serde_json::Value>, reasoning_effort: Option<&serde_json::Value>) -> Result<String> {
        let mut env = Environment::new();
        // HF chat templates are written for Python's Jinja2: they call str methods like .strip().
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        // Jinja2's `tojson` as Hugging Face templates use it: accepts sort_keys / separators / indent and does not
        // HTML-escape. (minijinja's built-in rejects those keyword arguments, which broke every request with tools.)
        env.add_filter("tojson", |v: Value, kw: Kwargs| -> Result<Value, Error> {
            // Without sort_keys, objects keep the order their keys arrived in, as in Python.
            let sort = kw.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
            let seps: Option<Vec<String>> = kw.get("separators")?;
            let indent: Option<usize> = kw.get("indent")?;
            kw.assert_all_used()?;
            let j = serde_json::to_value(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
            let (item, key) = match &seps {
                Some(s) if s.len() == 2 => (s[0].clone(), s[1].clone()),
                _ => (", ".to_string(), ": ".to_string()), // Python's defaults
            };
            let mut out = String::new();
            write_json(&j, &item, &key, indent, sort, 0, &mut out);
            Ok(Value::from_safe_string(out))
        });
        env.add_template("chat", &self.template)?;
        let out = env.get_template("chat")?.render(context! {
            messages => Self::canonicalize(messages),
            tools => tools,
            reasoning_effort => reasoning_effort,
            add_generation_prompt => true,
        })?;
        Ok(out)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Special tokens are already spelled out by the template, so don't add more.
        Ok(self.tokenizer.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }
}

/// JSON text with Python-style separators (`json.dumps(obj, separators=(item, key), indent=...)`), keys sorted.
fn write_json(v: &serde_json::Value, item: &str, key: &str, indent: Option<usize>, sort: bool, depth: usize, out: &mut String) {
    use serde_json::Value as J;
    let nl = |out: &mut String, d: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * d));
        }
    };
    let item = if indent.is_some() { item.trim_end() } else { item };
    match v {
        J::Array(a) if !a.is_empty() => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                nl(out, depth + 1);
                write_json(x, item, key, indent, sort, depth + 1, out);
            }
            nl(out, depth);
            out.push(']');
        }
        J::Object(m) if !m.is_empty() => {
            out.push('{');
            let mut entries: Vec<(&String, &J)> = m.iter().collect();
            if sort {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (k, x)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                nl(out, depth + 1);
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(key);
                write_json(x, item, key, indent, sort, depth + 1, out);
            }
            nl(out, depth);
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}
