use crate::{self as cubecl};
use alloc::{vec, vec::Vec};
use cubecl::prelude::*;
use cubecl_common::stream_id::StreamId;

#[cube(launch)]
pub fn big_task<F: Float>(input: &Array<u32>, output: &mut Array<F>, num_loop: usize) {
    if ABSOLUTE_POS > output.len() {
        terminate!()
    }

    for i in 0..num_loop {
        let pos = i % input.len();
        output[ABSOLUTE_POS] += F::cast_from(input[pos]) / F::cast_from(num_loop);
    }
}

pub fn test_stream_small<R: Runtime>(client: ComputeClient<R>) {
    test_stream_chained::<R, f32>(client, 32, 1, 32);
}

pub fn test_stream_medium<R: Runtime>(client: ComputeClient<R>) {
    test_stream_chained::<R, f32>(client, 256, 12, 256);
}

pub fn test_stream_chained<R: Runtime, F: Float + CubeElement>(
    client: ComputeClient<R>,
    len: usize,
    rounds: usize,
    num_loop: usize,
) {
    assert!(len > 0);
    assert!(num_loop > 0);
    assert_eq!(
        len % 32,
        0,
        "len must be divisible by 32 for the TT wrapper"
    );

    let client_1 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10000 });
        c
    };
    let client_2 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10001 });
        c
    };

    let input_seed: Vec<u32> = (0..len as u32).collect();
    let mut input = client_1.create_from_slice(u32::as_bytes(&input_seed));
    let mut output = None;

    let mut state: Vec<u32> = input_seed;
    let mut expected_first = 0.0f32;

    for _ in 0..rounds {
        let zero_output = vec![F::new(0.0); len];
        let output_ = client_1.create_from_slice(F::as_bytes(&zero_output));
        unsafe {
            big_task::launch::<F, R>(
                &client_1,
                CubeCount::Static(len as u32 / 32, 1, 1),
                CubeDim::new_1d(32),
                ArrayArg::from_raw_parts(input, len),
                ArrayArg::from_raw_parts(output_.clone(), len),
                num_loop,
            )
        };
        input = output_.clone();
        output = Some(output_);

        let total: f32 = (0..num_loop).map(|i| state[i % state.len()] as f32).sum();
        expected_first = total / num_loop as f32;
        state = vec![expected_first.to_bits(); len];
    }

    let actual = client_2.read_one_unchecked(output.unwrap());
    let actual = F::from_bytes(&actual);
    assert_eq!(actual[0], F::new(expected_first));
}

pub fn test_stream<R: Runtime, F: Float + CubeElement>(client: ComputeClient<R>) {
    let client_1 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10000 });
        c
    };
    let client_2 = unsafe {
        let mut c = client.clone();
        c.set_stream(StreamId { value: 10001 });
        c
    };

    let len = 4096;
    let input: Vec<u32> = (0..len as u32).collect();
    let mut input = client_1.create_from_slice(u32::as_bytes(&input));
    let mut output = None;

    for _ in 0..300 {
        let zero_output = vec![F::new(0.0); len];
        let output_ = client_1.create_from_slice(F::as_bytes(&zero_output));
        unsafe {
            big_task::launch::<F, R>(
                &client_1,
                CubeCount::Static(len as u32 / 32, 1, 1),
                CubeDim::new_1d(32),
                ArrayArg::from_raw_parts(input, len),
                ArrayArg::from_raw_parts(output_.clone(), len),
                4096,
            )
        };
        input = output_.clone();
        output = Some(output_);
    }

    let actual = client_2.read_one_unchecked(output.unwrap());
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(1318936000.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_stream {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        #[ignore = "Not yet supported by all backends"]
        fn test_stream() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::stream::test_stream::<TestRuntime, FloatType>(client);
        }
    };
}
