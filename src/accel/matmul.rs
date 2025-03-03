use nano_gemm::Plan;
use coe::is_same;
use num_traits::One;
use num_traits::Zero;

use crate::matrix::MatRef;
use crate::matrix::MatMut;
use crate::c32;
use crate::c64;
use crate::ComplexScalar;

/// Matrix-matrix multiplication.
pub fn matmul_unchecked<C: ComplexScalar>(lhs: MatRef<C>, rhs: MatRef<C>, out: MatMut<C>) {
    let m = lhs.nrows();
    let n = rhs.ncols();
    let k = lhs.ncols();

    if is_same::<C, c32>() {
        let plan = Plan::new_colmajor_lhs_and_dst_c32(m, n, k);
        let out: MatMut<c32> = unsafe { std::mem::transmute(out) };
        let rhs: MatRef<c32> = unsafe { std::mem::transmute(rhs) };
        let lhs: MatRef<c32> = unsafe { std::mem::transmute(lhs) };
        let out_col_stride = out.col_stride();
        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                c32::zero().into(),
                c32::one().into(),  // TODO: Figure if I can create custom kernels for one/zero alpha/beta
                false,
                false,
            );
        }
    } else {
        let plan = Plan::new_colmajor_lhs_and_dst_c64(m, n, k);
        let out: MatMut<c64> = unsafe { std::mem::transmute(out) };
        let rhs: MatRef<c64> = unsafe { std::mem::transmute(rhs) };
        let lhs: MatRef<c64> = unsafe { std::mem::transmute(lhs) };
        let out_col_stride = out.col_stride();
        unsafe {
            plan.execute_unchecked(
                m,
                n,
                k,
                out.as_ptr_mut() as _,
                1,
                out_col_stride,
                lhs.as_ptr() as _,
                1,
                lhs.col_stride(),
                rhs.as_ptr() as _,
                1,
                rhs.col_stride(),
                c64::zero().into(),
                c64::one().into(),  // TODO: Figure if I can create custom kernels for one/zero alpha/beta
                false,
                false,
            );
        }
    }
}


#[cfg(test)]
mod tests {
    // use crate::matrix::MatMut;
    // use crate::matrix::MatRef;
    use crate::matrix::Mat;
    use crate::c32;
    use num_traits::Zero;
    use super::matmul_unchecked;

    #[test]
    fn test_matmul_unchecked() {
        let m = 2;
        let n = 2;
        let k = 2;

        let mut lhs = Mat::<c32>::zeros(m, k);
        let mut rhs = Mat::<c32>::zeros(k, n);
        let mut out = Mat::<c32>::zeros(m, n);

        for i in 0..m {
            for j in 0..k {
                lhs[(i, j)] = c32::new((i + j) as f32, (i + j) as f32);
            }
        }

        for i in 0..k {
            for j in 0..n {
                rhs[(i, j)] = c32::new((i + j) as f32, (i + j) as f32);
            }
        }

        matmul_unchecked(
            lhs.as_ref(),
            rhs.as_ref(),
            out.as_mut(),
        );

        for i in 0..m {
            for j in 0..n {
                let mut sum = c32::zero();
                for l in 0..k {
                    sum += lhs[(i, l)] * rhs[(l, j)];
                }
                assert_eq!(out[(i, j)], sum);
            }
        }
    }
}
