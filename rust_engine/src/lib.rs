//! Karyon-CoRE High-Performance Rust Substrate Engine
//! Zero external dependencies, pure SIMD-friendly vector operations,
//! thread-safe associative memory matrix scans, and C-ABI export.

#[repr(C)]
pub struct RustStepMetrics {
    pub loss: f32,
    pub is_correct: bool,
    pub turns_taken: usize,
    pub beta: f32,
    pub s_norm: f32,
}

pub struct RustSovereignEngine {
    pub dim: usize,
    pub num_heads: usize,
    pub dim_k: usize,
    pub dim_v: usize,
    pub dim_emb: usize,
    pub s_memory: Vec<f32>,
}

impl RustSovereignEngine {
    pub fn new(dim: usize, num_heads: usize, dim_k: usize, dim_v: usize, dim_emb: usize) -> Self {
        let mem_size = num_heads * dim_k * dim_v;
        Self {
            dim,
            num_heads,
            dim_k,
            dim_v,
            dim_emb,
            s_memory: vec![0.0; mem_size],
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.s_memory.fill(0.0);
    }

    /// Fast memory trace readout: read_heads = q @ S
    #[inline(always)]
    pub fn associative_read(&self, q_heads: &[f32], read_out: &mut [f32]) {
        let d_k = self.dim_k;
        let d_v = self.dim_v;

        for h in 0..self.num_heads {
            let q_slice = &q_heads[h * d_k..(h + 1) * d_k];
            let out_slice = &mut read_out[h * d_v..(h + 1) * d_v];
            out_slice.fill(0.0);

            for k in 0..d_k {
                let q_val = q_slice[k];
                let mem_row = &self.s_memory[h * d_k * d_v + k * d_v..h * d_k * d_v + (k + 1) * d_v];
                for v in 0..d_v {
                    out_slice[v] += q_val * mem_row[v];
                }
            }
        }
    }

    /// Fast outer-product Hebbian/Delta write into S_memory
    #[inline(always)]
    pub fn associative_write(&mut self, k_heads: &[f32], v_heads: &[f32], alpha: f32, eta: f32) {
        let d_k = self.dim_k;
        let d_v = self.dim_v;

        for h in 0..self.num_heads {
            let k_slice = &k_heads[h * d_k..(h + 1) * d_k];
            let v_slice = &v_heads[h * d_v..(h + 1) * d_v];

            for k in 0..d_k {
                let k_val = k_slice[k];
                let offset = h * d_k * d_v + k * d_v;
                for v in 0..d_v {
                    let old = self.s_memory[offset + v];
                    self.s_memory[offset + v] = alpha * old + eta * k_val * v_slice[v];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// C-ABI Safe Exports for C, C++, and Python FFI
// ----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn karyon_rust_create(
    dim: usize,
    num_heads: usize,
    dim_k: usize,
    dim_v: usize,
    dim_emb: usize,
) -> *mut RustSovereignEngine {
    Box::into_raw(Box::new(RustSovereignEngine::new(
        dim, num_heads, dim_k, dim_v, dim_emb,
    )))
}

#[no_mangle]
pub extern "C" fn karyon_rust_destroy(engine: *mut RustSovereignEngine) {
    if !engine.is_null() {
        unsafe {
            drop(Box::from_raw(engine));
        }
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_reset(engine: *mut RustSovereignEngine) {
    if let Some(eng) = unsafe { engine.as_mut() } {
        eng.reset();
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_read(
    engine: *const RustSovereignEngine,
    q_heads: *const f32,
    read_out: *mut f32,
) {
    if let Some(eng) = unsafe { engine.as_ref() } {
        let q_len = eng.num_heads * eng.dim_k;
        let read_len = eng.num_heads * eng.dim_v;
        let q_slice = unsafe { std::slice::from_raw_parts(q_heads, q_len) };
        let read_slice = unsafe { std::slice::from_raw_parts_mut(read_out, read_len) };
        eng.associative_read(q_slice, read_slice);
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_write(
    engine: *mut RustSovereignEngine,
    k_heads: *const f32,
    v_heads: *const f32,
    alpha: f32,
    eta: f32,
) {
    if let Some(eng) = unsafe { engine.as_mut() } {
        let k_len = eng.num_heads * eng.dim_k;
        let v_len = eng.num_heads * eng.dim_v;
        let k_slice = unsafe { std::slice::from_raw_parts(k_heads, k_len) };
        let v_slice = unsafe { std::slice::from_raw_parts(v_heads, v_len) };
        eng.associative_write(k_slice, v_slice, alpha, eta);
    }
}
