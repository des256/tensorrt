fn main() {
    if std::env::var("CARGO_FEATURE_TRT").is_ok() {
        cc::Build::new()
            .cuda(true)
            .file("src/kernels/argmax.cu")
            .include("/usr/local/cuda/include")
            .compile("parakeet_kernels");

        println!("cargo:rerun-if-changed=src/kernels/argmax.cu");
    }
}
