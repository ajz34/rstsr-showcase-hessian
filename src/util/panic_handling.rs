pub(crate) trait NIIntoUsizeVec {
    fn into_usize_vec(self) -> Vec<usize>;
}

impl NIIntoUsizeVec for usize {
    fn into_usize_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl NIIntoUsizeVec for &[usize] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> NIIntoUsizeVec for [usize; N] {
    fn into_usize_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl NIIntoUsizeVec for Vec<usize> {
    fn into_usize_vec(self) -> Vec<usize> {
        self
    }
}

impl NIIntoUsizeVec for &Vec<usize> {
    fn into_usize_vec(self) -> Vec<usize> {
        self.clone()
    }
}

#[macro_export]
macro_rules! check_shape {
    ($actual:expr, $expected:expr, $msg:expr) => {{
        if $actual.into_usize_vec() != $expected.into_usize_vec() {
            let str_actual = stringify!($actual);
            let str_expected = stringify!($expected);
            panic!(
                "Shape mismatch: expected {} = {:?}, but got {} = {:?}; message: {}",
                str_expected,
                $expected.into_usize_vec(),
                str_actual,
                $actual.into_usize_vec(),
                $msg
            );
        }
    }};

    ($cond:expr, $msg:expr) => {{
        if !$cond {
            let str_cond = stringify!($cond);
            panic!("Condition failed: {}; message: {}", str_cond, $msg);
        }
    }};
}
