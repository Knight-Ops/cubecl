use crate as cubecl;
use alloc::vec::Vec;

use cubecl::prelude::*;

#[cube(launch, address_type = "dynamic")]
pub fn kernel_absolute_pos(output1: &mut Array<u32>) {
    if ABSOLUTE_POS >= output1.len() {
        terminate!();
    }

    output1[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
}

#[cube(launch, address_type = "dynamic")]
pub fn kernel_axis_components(
    absolute_pos_x: &mut Array<u32>,
    cube_pos_x: &mut Array<u32>,
    unit_pos_x: &mut Array<u32>,
) {
    if ABSOLUTE_POS >= absolute_pos_x.len() {
        terminate!();
    }

    absolute_pos_x[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
    cube_pos_x[ABSOLUTE_POS] = CUBE_POS_X as u32;
    unit_pos_x[ABSOLUTE_POS] = UNIT_POS_X as u32;
}

#[cube(launch, address_type = "dynamic")]
pub fn kernel_axis_components_2d(
    absolute_pos_x: &mut Array<u32>,
    unit_pos_x: &mut Array<u32>,
    unit_pos_y: &mut Array<u32>,
) {
    if ABSOLUTE_POS >= absolute_pos_x.len() {
        terminate!();
    }

    absolute_pos_x[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
    unit_pos_x[ABSOLUTE_POS] = UNIT_POS_X as u32;
    unit_pos_y[ABSOLUTE_POS] = UNIT_POS_Y as u32;
}

#[cube(launch, address_type = "dynamic")]
pub fn kernel_cube_components_3d(
    cube_pos_x_out: &mut Array<u32>,
    cube_pos_y_out: &mut Array<u32>,
    cube_pos_z_out: &mut Array<u32>,
) {
    if ABSOLUTE_POS >= cube_pos_x_out.len() {
        terminate!();
    }

    cube_pos_x_out[ABSOLUTE_POS] = CUBE_POS_X as u32;
    cube_pos_y_out[ABSOLUTE_POS] = CUBE_POS_Y as u32;
    cube_pos_z_out[ABSOLUTE_POS] = CUBE_POS_Z as u32;
}

fn run_topology_axis_components_case<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
    cube_count: (u32, u32, u32),
    cube_dim: (u32, u32, u32),
    length: u32,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let byte_len = length as usize * core::mem::size_of::<u32>();
    let absolute_handle = client.empty(byte_len);
    let cube_handle = client.empty(byte_len);
    let unit_handle = client.empty(byte_len);

    unsafe {
        kernel_axis_components::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(absolute_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(cube_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(unit_handle.clone(), length as usize),
        )
    };

    let actual_absolute_bytes = client.read_one_unchecked(absolute_handle);
    let actual_cube_bytes = client.read_one_unchecked(cube_handle);
    let actual_unit_bytes = client.read_one_unchecked(unit_handle);
    let actual_absolute = u32::from_bytes(&actual_absolute_bytes);
    let actual_cube = u32::from_bytes(&actual_cube_bytes);
    let actual_unit = u32::from_bytes(&actual_unit_bytes);

    let expect_absolute: Vec<u32> = (0..length).collect();
    let expect_cube: Vec<u32> = (0..length).map(|index| index / cube_dim.0).collect();
    let expect_unit: Vec<u32> = (0..length).map(|index| index % cube_dim.0).collect();

    assert_eq!(actual_absolute, &expect_absolute);
    assert_eq!(actual_cube, &expect_cube);
    assert_eq!(actual_unit, &expect_unit);
}


fn run_topology_axis_components_2d_case<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
    cube_dim: (u32, u32, u32),
    length: u32,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let byte_len = length as usize * core::mem::size_of::<u32>();
    let absolute_handle = client.empty(byte_len);
    let unit_x_handle = client.empty(byte_len);
    let unit_y_handle = client.empty(byte_len);

    unsafe {
        kernel_axis_components_2d::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(absolute_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(unit_x_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(unit_y_handle.clone(), length as usize),
        )
    };

    let actual_absolute_bytes = client.read_one_unchecked(absolute_handle);
    let actual_unit_x_bytes = client.read_one_unchecked(unit_x_handle);
    let actual_unit_y_bytes = client.read_one_unchecked(unit_y_handle);
    let actual_absolute = u32::from_bytes(&actual_absolute_bytes);
    let actual_unit_x = u32::from_bytes(&actual_unit_x_bytes);
    let actual_unit_y = u32::from_bytes(&actual_unit_y_bytes);

    let expect_absolute: Vec<u32> = (0..length).collect();
    let expect_unit_x: Vec<u32> = (0..length).map(|index| index % cube_dim.0).collect();
    let expect_unit_y: Vec<u32> = (0..length)
        .map(|index| (index / cube_dim.0) % cube_dim.1)
        .collect();

    assert_eq!(actual_absolute, &expect_absolute);
    assert_eq!(actual_unit_x, &expect_unit_x);
    assert_eq!(actual_unit_y, &expect_unit_y);
}

pub fn test_kernel_topology_absolute_pos<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let cube_count = (3, 5, 7);
    let cube_dim = (16, 16, 1);

    let length = cube_count.0 * cube_count.1 * cube_count.2 * cube_dim.0 * cube_dim.1 * cube_dim.2;
    let handle1 = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(handle1.clone(), length as usize),
        )
    };

    let actual = client.read_one_unchecked(handle1);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..length).collect();

    assert_eq!(actual, &expect);
}

pub fn test_kernel_topology_absolute_pos_linearized<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    // Keep TT topology bring-up honest: this exercises absolute-position
    // indexing across many logical cubes while staying within the current
    // single-axis cube-dimension model.
    let cube_count = (105, 1, 1);
    let cube_dim = (256, 1, 1);

    let length = cube_count.0 * cube_count.1 * cube_count.2 * cube_dim.0 * cube_dim.1 * cube_dim.2;
    let handle1 = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(handle1.clone(), length as usize),
        )
    };

    let actual = client.read_one_unchecked(handle1);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..length).collect();

    assert_eq!(actual, &expect);
}

pub fn test_kernel_topology_axis_components_linearized<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    // Stay within TT's current topology model while still validating that
    // per-axis topology builtins line up with the linearized launch layout.
    run_topology_axis_components_case(client, addr_type, (105, 1, 1), (256, 1, 1), 105 * 256);
}

pub fn test_kernel_topology_axis_components_linearized_tail<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    // Exercise the same single-axis decomposition with a non-full tail so TT
    // wrappers can validate bounds handling without relying on y/z semantics.
    run_topology_axis_components_case(client, addr_type, (5, 1, 1), (64, 1, 1), 5 * 64 - 13);
}

pub fn test_kernel_topology_axis_components_2d_single_cube<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    // Keep TT non-1D bring-up honest: this validates 2D unit-position
    // decomposition within a single logical cube-count lane before broader
    // multidimensional cube-count semantics are claimed.
    run_topology_axis_components_2d_case(client, addr_type, (12, 3, 1), 12 * 3);
}

pub fn test_kernel_topology_axis_components_2d_single_cube_tail<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    run_topology_axis_components_2d_case(client, addr_type, (12, 3, 1), 12 * 3 - 5);
}


pub fn test_kernel_topology_cube_components_3d<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let cube_count = (3, 5, 7);
    let cube_dim = (4, 2, 1);
    let length = cube_count.0 * cube_count.1 * cube_count.2 * cube_dim.0 * cube_dim.1 * cube_dim.2;
    let byte_len = length as usize * core::mem::size_of::<u32>();
    let cube_x_handle = client.empty(byte_len);
    let cube_y_handle = client.empty(byte_len);
    let cube_z_handle = client.empty(byte_len);

    unsafe {
        kernel_cube_components_3d::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            CubeDim {
                x: cube_dim.0,
                y: cube_dim.1,
                z: cube_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(cube_x_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(cube_y_handle.clone(), length as usize),
            ArrayArg::from_raw_parts(cube_z_handle.clone(), length as usize),
        )
    };

    let actual_cube_x_bytes = client.read_one_unchecked(cube_x_handle);
    let actual_cube_y_bytes = client.read_one_unchecked(cube_y_handle);
    let actual_cube_z_bytes = client.read_one_unchecked(cube_z_handle);
    let actual_cube_x = u32::from_bytes(&actual_cube_x_bytes);
    let actual_cube_y = u32::from_bytes(&actual_cube_y_bytes);
    let actual_cube_z = u32::from_bytes(&actual_cube_z_bytes);

    let cube_units = cube_dim.0 * cube_dim.1 * cube_dim.2;
    let expect_cube_x: Vec<u32> = (0..length)
        .map(|index| (index / cube_units) % cube_count.0)
        .collect();
    let expect_cube_y: Vec<u32> = (0..length)
        .map(|index| ((index / cube_units) / cube_count.0) % cube_count.1)
        .collect();
    let expect_cube_z: Vec<u32> = (0..length)
        .map(|index| (index / cube_units) / (cube_count.0 * cube_count.1))
        .collect();

    assert_eq!(actual_cube_x, &expect_cube_x);
    assert_eq!(actual_cube_y, &expect_cube_y);
    assert_eq!(actual_cube_z, &expect_cube_z);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_topology {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_topology_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos::<TestRuntime>(
                client.clone(),
                AddressType::U32,
            );
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos::<TestRuntime>(
                client,
                AddressType::U64,
            );
        }
    };
}
