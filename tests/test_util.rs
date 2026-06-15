#![allow(dead_code)]

use rayon::prelude::*;
use rstsr::prelude::*;

use npyz::NpyFile;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rstsr_showcase_hessian::prelude_dev::DeviceTsr;

pub type Tsr<T = f64> = Tensor<T, DeviceTsr, IxD>;
pub type TsrView<'a, T = f64> = TensorView<'a, T, DeviceTsr, IxD>;

/// Read a tensor from npz file in prototype directory.
///
/// Note the returned tensor is always in row-major order (which should be according to NumPy's
/// convention). However, the shape is usually not the same to our convention, and some
/// transposition may be needed.
pub fn read_npz(file: &str, name: &str) -> Tsr {
    let cargo_manifest_path = std::env!("CARGO_MANIFEST_DIR");
    let path = Path::new(cargo_manifest_path).join("prototype").join(file);
    let npz_file = BufReader::new(File::open(path).unwrap());
    let mut zip_file = zip::ZipArchive::new(npz_file).unwrap();
    let npy_file = zip_file.by_name(&format!("{name}.npy")).unwrap();
    let npy_reader = NpyFile::new(npy_file).unwrap();
    let shape = npy_reader.shape().iter().map(|&dim| dim as usize).collect::<Vec<_>>();
    let data = npy_reader.into_vec::<f64>().unwrap();
    let device = DeviceTsr::default();
    rt::asarray((data, shape.c(), &device))
}

/// Read all tensors from npz file in prototype directory, and return as a dictionary.
pub fn read_npz_dict(file: &str) -> HashMap<String, Tsr> {
    let cargo_manifest_path = std::env!("CARGO_MANIFEST_DIR");
    let path = Path::new(cargo_manifest_path).join("prototype").join(file);
    let npz_file = BufReader::new(File::open(path).unwrap());
    let mut zip_file = zip::ZipArchive::new(npz_file).unwrap();
    let mut dict = HashMap::new();
    for i in 0..zip_file.len() {
        let file = zip_file.by_index(i).unwrap();
        if file.name().ends_with(".npy") {
            let name = file.name().trim_end_matches(".npy").to_string();
            let npy_reader = NpyFile::new(file).unwrap();
            let shape = npy_reader.shape().iter().map(|&dim| dim as usize).collect::<Vec<_>>();
            let data = npy_reader.into_vec::<f64>().unwrap();
            let device = DeviceTsr::default();
            dict.insert(name, rt::asarray((data, shape.c(), &device)));
        }
    }
    dict
}

/// A simple fingerprint function (like `pyscf.lib.fp`) for testing.
///
/// Note this function requires column-major order iteration. This is different to PySCF's row-major
/// order.
pub fn fp(x: TsrView) -> f64 {
    x.iter_with_order(TensorIterOrder::F).into_par_iter().enumerate().map(|(i, &v)| (i as f64).cos() * v).sum()
}
