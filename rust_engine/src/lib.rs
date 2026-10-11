//! Karyon-CoRE High-Performance Rust Substrate Engine
//! EXP-456 Sovereign Master: Multi-Head Dynamic Law Synthesizer & Dissipative Memory
//! Zero external dependencies, pure SIMD-friendly native safe/unsafe Rust.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustStepMetrics {
    pub loss: f32,
    pub pred_byte: u8,
    pub is_correct: bool,
    pub turns_taken: usize,
    pub entropy: f32,
    pub beta: f32,
    pub mean_alpha: f32,
    pub mean_eta_delta: f32,
    pub mean_eta_hebb: f32,
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

    /// Optimized memory trace readout: read_heads = q @ S
    #[inline(always)]
    pub fn associative_read(&self, q_heads: &[f32], read_out: &mut [f32]) {
        let d_k = self.dim_k;
        let d_v = self.dim_v;

        for h in 0..self.num_heads {
            let q_slice = &q_heads[h * d_k..(h + 1) * d_k];
            let out_slice = &mut read_out[h * d_v..(h + 1) * d_v];
            out_slice.fill(0.0);

            let mem_head = &self.s_memory[h * d_k * d_v..(h + 1) * d_k * d_v];

            for k in 0..d_k {
                let q_val = q_slice[k];
                if q_val == 0.0 { continue; }
                let mem_row = &mem_head[k * d_v..(k + 1) * d_v];
                
                for v in 0..d_v {
                    out_slice[v] += q_val * mem_row[v];
                }
            }
        }
    }

    /// Multi-Head Sovereign Dynamic Law Application:
    /// Applies synthesized (alpha, eta_delta, eta_hebb, lambda) per head directly into S_memory.
    #[inline(always)]
    pub fn apply_synthesized_laws(
        &mut self,
        k_heads: &[f32],
        v_heads: &[f32],
        v_pred_heads: &[f32],
        alphas: &[f32],
        etas_delta: &[f32],
        etas_hebb: &[f32],
        lambdas: &[f32],
    ) {
        let d_k = self.dim_k;
        let d_v = self.dim_v;

        for h in 0..self.num_heads {
            let alpha = alphas[h];
            let eta_d = etas_delta[h];
            let eta_h = etas_hebb[h];
            let lam = lambdas[h];

            let k_slice = &k_heads[h * d_k..(h + 1) * d_k];
            let v_slice = &v_heads[h * d_v..(h + 1) * d_v];
            let v_pred_slice = &v_pred_heads[h * d_v..(h + 1) * d_v];
            let mem_offset = h * d_k * d_v;

            for k in 0..d_k {
                let k_val = k_slice[k];
                let offset = mem_offset + k * d_v;
                let mem_row = &mut self.s_memory[offset..offset + d_v];

                for v in 0..d_v {
                    let err = v_slice[v] - v_pred_slice[v];
                    mem_row[v] = alpha * mem_row[v] 
                               + eta_d * (k_val * err) 
                               + eta_h * (k_val * v_slice[v]) 
                               - lam * mem_row[v];
                }
            }
        }

        // Bounded Norm Clamping (Tononi SHY / Thermodynamic Stability)
        let mut total_norm_sq = 0.0f32;
        for val in &self.s_memory {
            total_norm_sq += val * val;
        }
        let total_norm = total_norm_sq.sqrt();
        if total_norm > 30.0 {
            let scale = 30.0 / total_norm;
            for val in &mut self.s_memory {
                *val *= scale;
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
    let engine = Box::new(RustSovereignEngine::new(dim, num_heads, dim_k, dim_v, dim_emb));
    Box::into_raw(engine)
}

#[no_mangle]
pub extern "C" fn karyon_rust_destroy(engine_ptr: *mut RustSovereignEngine) {
    if !engine_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(engine_ptr));
        }
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_reset(engine_ptr: *mut RustSovereignEngine) {
    if let Some(engine) = unsafe { engine_ptr.as_mut() } {
        engine.reset();
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_associative_read(
    engine_ptr: *const RustSovereignEngine,
    q_heads_ptr: *const f32,
    read_out_ptr: *mut f32,
) {
    if let Some(engine) = unsafe { engine_ptr.as_ref() } {
        let q_len = engine.num_heads * engine.dim_k;
        let out_len = engine.num_heads * engine.dim_v;
        let q_slice = unsafe { std::slice::from_raw_parts(q_heads_ptr, q_len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(read_out_ptr, out_len) };
        engine.associative_read(q_slice, out_slice);
    }
}

#[no_mangle]
pub extern "C" fn karyon_rust_apply_laws(
    engine_ptr: *mut RustSovereignEngine,
    k_heads_ptr: *const f32,
    v_heads_ptr: *const f32,
    v_pred_heads_ptr: *const f32,
    alphas_ptr: *const f32,
    etas_delta_ptr: *const f32,
    etas_hebb_ptr: *const f32,
    lambdas_ptr: *const f32,
) {
    if let Some(engine) = unsafe { engine_ptr.as_mut() } {
        let k_len = engine.num_heads * engine.dim_k;
        let v_len = engine.num_heads * engine.dim_v;
        let h_len = engine.num_heads;

        let k_slice = unsafe { std::slice::from_raw_parts(k_heads_ptr, k_len) };
        let v_slice = unsafe { std::slice::from_raw_parts(v_heads_ptr, v_len) };
        let v_pred_slice = unsafe { std::slice::from_raw_parts(v_pred_heads_ptr, v_len) };
        let alphas = unsafe { std::slice::from_raw_parts(alphas_ptr, h_len) };
        let etas_delta = unsafe { std::slice::from_raw_parts(etas_delta_ptr, h_len) };
        let etas_hebb = unsafe { std::slice::from_raw_parts(etas_hebb_ptr, h_len) };
        let lambdas = unsafe { std::slice::from_raw_parts(lambdas_ptr, h_len) };

        engine.apply_synthesized_laws(
            k_slice,
            v_slice,
            v_pred_slice,
            alphas,
            etas_delta,
            etas_hebb,
            lambdas,
        );
    }
}
