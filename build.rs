fn main() {
    #[cfg(feature = "openblas")]
    {
        println!("cargo:rustc-link-lib=dylib=openblas");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}
