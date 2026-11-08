//! PyO3 bindings, built by maturin into the `ad_auction_lab_rs` extension module.
//!
//! Arrays cross the boundary through the buffer protocol and are read in place (they must be
//! C-contiguous numpy arrays of the documented dtype); outputs are returned as bytes for
//! `numpy.frombuffer`. The GIL is released around every heavy loop.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::fm::{Fm, FmParams};
use crate::ftrl::{Ftrl, FtrlParams, predict_linear};
use crate::hashing::Features;
use crate::pacing::{Market, Pacer, simulate};

fn slice<'a, T: Element>(buf: &'a PyBuffer<T>, name: &str) -> PyResult<&'a [T]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    // SAFETY: the buffer is C-contiguous with element type T (checked by PyBuffer::get) and stays
    // alive and exported for as long as `buf` is borrowed.
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const T, buf.item_count()) })
}

fn bytes_of<'py, T: Copy>(py: Python<'py>, values: &[T]) -> Bound<'py, PyBytes> {
    // SAFETY: plain-old-data numeric slices viewed as bytes.
    let raw = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    PyBytes::new(py, raw)
}

fn copy_into(dst: &mut [f32], src: &PyBuffer<f32>, name: &str) -> PyResult<()> {
    let s = slice(src, name)?;
    if s.len() != dst.len() {
        return Err(PyValueError::new_err(format!(
            "{name} has {} values, expected {}",
            s.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(s);
    Ok(())
}

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

struct Inputs {
    small: PyBuffer<u8>,
    n_small: usize,
    wide: PyBuffer<u32>,
    n_wide: usize,
    rows: PyBuffer<u64>,
}

impl Inputs {
    fn features(&self) -> PyResult<Features<'_>> {
        Features::new(
            slice(&self.small, "small")?,
            self.n_small,
            slice(&self.wide, "wide")?,
            self.n_wide,
        )
        .map_err(err)
    }
}

/// FTRL-Proximal logistic regression on hashed features.
#[pyclass(module = "ad_auction_lab.ad_auction_lab_rs")]
struct FtrlModel {
    inner: Ftrl,
}

#[pymethods]
impl FtrlModel {
    #[new]
    fn new(alpha: f64, beta: f64, l1: f64, l2: f64, bits: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Ftrl::new(FtrlParams {
                alpha,
                beta,
                l1,
                l2,
                bits,
            })
            .map_err(err)?,
        })
    }

    /// One pass over `rows`; returns the progressive mean log loss.
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        small: PyBuffer<u8>,
        n_small: usize,
        wide: PyBuffer<u32>,
        n_wide: usize,
        labels: PyBuffer<u8>,
        rows: PyBuffer<u64>,
    ) -> PyResult<f64> {
        let inp = Inputs {
            small,
            n_small,
            wide,
            n_wide,
            rows,
        };
        let x = inp.features()?;
        let y = slice(&labels, "labels")?;
        let r = slice(&inp.rows, "rows")?;
        let inner = &mut self.inner;
        py.detach(|| inner.fit(&x, y, r)).map_err(err)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        small: PyBuffer<u8>,
        n_small: usize,
        wide: PyBuffer<u32>,
        n_wide: usize,
        rows: PyBuffer<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let inp = Inputs {
            small,
            n_small,
            wide,
            n_wide,
            rows,
        };
        let x = inp.features()?;
        let r = slice(&inp.rows, "rows")?;
        let p = py.detach(|| self.inner.predict(&x, r)).map_err(err)?;
        Ok(bytes_of(py, &p))
    }

    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let w = py.detach(|| self.inner.weights());
        bytes_of(py, &w)
    }

    fn get_z<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        bytes_of(py, &self.inner.z)
    }

    fn get_n<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        bytes_of(py, &self.inner.n)
    }

    fn set_state(&mut self, z: PyBuffer<f32>, n: PyBuffer<f32>) -> PyResult<()> {
        copy_into(&mut self.inner.z, &z, "z")?;
        copy_into(&mut self.inner.n, &n, "n")
    }
}

/// Probabilities of a hashed linear model with weights `w` (`2^bits + 1` values, bias last).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn predict_linear_weights<'py>(
    py: Python<'py>,
    w: PyBuffer<f32>,
    bits: u32,
    small: PyBuffer<u8>,
    n_small: usize,
    wide: PyBuffer<u32>,
    n_wide: usize,
    rows: PyBuffer<u64>,
) -> PyResult<Bound<'py, PyBytes>> {
    if !(1..=30).contains(&bits) {
        return Err(PyValueError::new_err("bits must be between 1 and 30"));
    }
    let w = slice(&w, "w")?;
    if w.len() != (1usize << bits) + 1 {
        return Err(PyValueError::new_err("w must have 2^bits + 1 values"));
    }
    let inp = Inputs {
        small,
        n_small,
        wide,
        n_wide,
        rows,
    };
    let x = inp.features()?;
    let r = slice(&inp.rows, "rows")?;
    let mask = (1u64 << bits) - 1;
    let p = py.detach(|| predict_linear(w, mask, &x, r)).map_err(err)?;
    Ok(bytes_of(py, &p))
}

/// Factorization machine on hashed features.
#[pyclass(module = "ad_auction_lab.ad_auction_lab_rs")]
struct FmModel {
    inner: Fm,
}

#[pymethods]
impl FmModel {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k: usize,
        bits: u32,
        lr: f64,
        l2_linear: f64,
        l2_factor: f64,
        init_scale: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let p = FmParams {
            k,
            bits,
            lr,
            l2_linear,
            l2_factor,
            init_scale,
            seed,
        };
        Ok(Self {
            inner: Fm::new(p).map_err(err)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        small: PyBuffer<u8>,
        n_small: usize,
        wide: PyBuffer<u32>,
        n_wide: usize,
        labels: PyBuffer<u8>,
        rows: PyBuffer<u64>,
    ) -> PyResult<f64> {
        let inp = Inputs {
            small,
            n_small,
            wide,
            n_wide,
            rows,
        };
        let x = inp.features()?;
        let y = slice(&labels, "labels")?;
        let r = slice(&inp.rows, "rows")?;
        let inner = &mut self.inner;
        py.detach(|| inner.fit(&x, y, r)).map_err(err)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        small: PyBuffer<u8>,
        n_small: usize,
        wide: PyBuffer<u32>,
        n_wide: usize,
        rows: PyBuffer<u64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let inp = Inputs {
            small,
            n_small,
            wide,
            n_wide,
            rows,
        };
        let x = inp.features()?;
        let r = slice(&inp.rows, "rows")?;
        let p = py.detach(|| self.inner.predict(&x, r)).map_err(err)?;
        Ok(bytes_of(py, &p))
    }

    /// Model state as a dict of bytes (`w0`, `g0` as floats).
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("w0", self.inner.w0)?;
        d.set_item("g0", self.inner.g0)?;
        d.set_item("w", bytes_of(py, &self.inner.w))?;
        d.set_item("gw", bytes_of(py, &self.inner.gw))?;
        d.set_item("v", bytes_of(py, &self.inner.v))?;
        d.set_item("gv", bytes_of(py, &self.inner.gv))?;
        Ok(d)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_state(
        &mut self,
        w0: f64,
        g0: f64,
        w: PyBuffer<f32>,
        gw: PyBuffer<f32>,
        v: PyBuffer<f32>,
        gv: PyBuffer<f32>,
    ) -> PyResult<()> {
        self.inner.w0 = w0;
        self.inner.g0 = g0;
        copy_into(&mut self.inner.w, &w, "w")?;
        copy_into(&mut self.inner.gw, &gw, "gw")?;
        copy_into(&mut self.inner.v, &v, "v")?;
        copy_into(&mut self.inner.gv, &gv, "gv")
    }
}

/// Run the pacing simulator. `pacer` is "none", "pid" or "dual"; returns a dict of results.
#[pyfunction]
#[pyo3(signature = (values, budgets, ctr, market, reserve, pacer, checkpoints=24, kp=0.0, ki=0.0, kd=0.0, interval=1, m_min=0.0, step=0.0))]
#[allow(clippy::too_many_arguments)]
fn simulate_pacing<'py>(
    py: Python<'py>,
    values: PyBuffer<f64>,
    budgets: PyBuffer<f64>,
    ctr: PyBuffer<f32>,
    market: PyBuffer<f32>,
    reserve: f64,
    pacer: &str,
    checkpoints: usize,
    kp: f64,
    ki: f64,
    kd: f64,
    interval: usize,
    m_min: f64,
    step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = match pacer {
        "none" => Pacer::None,
        "pid" => Pacer::Pid {
            kp,
            ki,
            kd,
            interval,
            m_min,
        },
        "dual" => Pacer::Dual { step },
        other => return Err(PyValueError::new_err(format!("unknown pacer {other:?}"))),
    };
    let m = Market {
        values: slice(&values, "values")?,
        budgets: slice(&budgets, "budgets")?,
        ctr: slice(&ctr, "ctr")?,
        market: slice(&market, "market")?,
        reserve,
    };
    let o = py.detach(|| simulate(&m, p, checkpoints)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("value", o.value)?;
    d.set_item("spend", o.spend)?;
    d.set_item("wins", o.wins)?;
    d.set_item("exhausted_at", o.exhausted_at)?;
    d.set_item("trajectory", bytes_of(py, &o.trajectory))?;
    d.set_item("revenue", o.revenue)?;
    d.set_item("market_wins", o.market_wins)?;
    d.set_item("unsold", o.unsold)?;
    Ok(d)
}

#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn ad_auction_lab_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FtrlModel>()?;
    m.add_class::<FmModel>()?;
    m.add_function(wrap_pyfunction!(predict_linear_weights, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_pacing, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
