//! Zero-copy reader for .safetensors files.
//!
//! Layout: 8 bytes little-endian header length N, then N bytes of JSON describing every
//! tensor (dtype, shape, byte range), then the raw tensor bytes. We mmap the file so the
//! 19.5 GB checkpoint costs no RAM until a tensor's pages are actually touched.
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::{collections::BTreeMap, fs::File, path::Path};

#[derive(Debug, Deserialize, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

pub struct SafeTensors {
    mmap: Mmap,
    data_start: usize,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the checkpoint is read-only on disk and we never write through the map.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 8 {
            bail!("{}: too short to be safetensors", path.display());
        }
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let data_start = 8 + n;
        if data_start > mmap.len() {
            bail!("{}: header length {} exceeds file size", path.display(), n);
        }
        let mut raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..data_start])?;
        raw.remove("__metadata__");
        let mut tensors = BTreeMap::new();
        for (name, v) in raw {
            let info: TensorInfo = serde_json::from_value(v).with_context(|| format!("tensor {name}"))?;
            if data_start + info.data_offsets[1] > mmap.len() {
                bail!("tensor {name} runs past end of file");
            }
            tensors.insert(name, info);
        }
        Ok(Self { mmap, data_start, tensors })
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensors.get(name).with_context(|| format!("no tensor named {name}"))?;
        Ok(&self.mmap[self.data_start + t.data_offsets[0]..self.data_start + t.data_offsets[1]])
    }
}
