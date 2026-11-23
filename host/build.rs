// Compile the CUDA engine with nvcc and link it into the Rust binary.
// Target sm_120a (not plain sm_120): the 'a' variant enables the block-scaled FP4 tensor-core instruction.
use std::{env, path::PathBuf, process::Command};

/// The CUDA toolkit: $CUDA_HOME, else the project-local one made by install.sh (pip wheels, no root), else /usr/local/cuda.
fn cuda_home() -> String {
    if let Ok(c) = env::var("CUDA_HOME") {
        return c;
    }
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    if let Ok(dirs) = std::fs::read_dir(root.join(".toolchain/lib")) {
        for d in dirs.flatten() {
            let c = d.path().join("site-packages/nvidia/cu13");
            if c.join("bin/nvcc").exists() {
                return c.canonicalize().unwrap().display().to_string();
            }
        }
    }
    if PathBuf::from("/usr/local/cuda/bin/nvcc").exists() {
        return "/usr/local/cuda".into();
    }
    panic!("no CUDA toolkit found: run ./install.sh, or set CUDA_HOME to a CUDA 13 (or 12.8+) toolkit");
}

fn main() {
    let cuda = cuda_home();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = "../engine/engine.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let obj = out.join("engine.o");
    let ok = Command::new(format!("{cuda}/bin/nvcc"))
        // --fmad=false: the model spec requires several multiply-then-add expressions NOT to be contracted into
        // fused multiply-adds; explicit fmaf() calls in the kernels are unaffected.
        .args(["-O3", "--fmad=false", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xcompiler", "-fPIC", "-c", src, "-o"])
        .arg(&obj)
        .status()
        .expect("failed to run nvcc")
        .success();
    assert!(ok, "nvcc failed");
    let lib = out.join("libinkengine.a");
    let _ = std::fs::remove_file(&lib);
    assert!(Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=inkengine");
    // Link the CUDA runtime by its exact file name: the pip-packaged toolkit has no unversioned libcudart.so.
    let libdir = ["lib", "lib64"].iter().map(|l| PathBuf::from(&cuda).join(l)).find(|d| d.exists()).expect("CUDA lib directory");
    let mut rt: Vec<String> = std::fs::read_dir(&libdir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("libcudart.so.") && n.matches('.').count() == 2).collect();
    rt.sort();
    let rt = rt.pop().expect("libcudart.so.<major> not found in the CUDA lib directory");
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-arg=-l:{rt}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
    // The driver library (installed with the NVIDIA driver): virtual memory management for the on-demand context caches.
    println!("cargo:rustc-link-arg=-l:libcuda.so.1");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
