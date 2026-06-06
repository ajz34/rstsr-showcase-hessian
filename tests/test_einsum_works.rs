use rstsr::prelude::*;

#[test]
fn test_einsum_works() {
    let device = DeviceFaer::default(); // any device with rayon support can be used.
    let (nao, nmo): (usize, usize) = (3, 2);
    let c = rt::arange(((nao * nmo) as f64, &device)).into_shape((nao, nmo));
    let e = rt::arange(((nao * nao * nao * nao) as f64, &device)).into_shape((nao, nao, nao, nao));

    let g = rt::tblis::einsum(
        "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
        [&c, &c, &e, &c, &c],     // tensors to be contracted
        true,                     // contraction strategy (see crate opt-einsum-path)
        None,                     // memory limit (None means no limit, see crate opt-einsum-path)
    );
    println!("{:?}", g);
}
