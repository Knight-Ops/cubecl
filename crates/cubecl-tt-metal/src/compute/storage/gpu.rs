use cubecl_common::{backtrace::BackTrace, stream_id::StreamId};
use cubecl_runtime::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::collections::HashMap;

use libtt_metal_cxx::MeshBuffer;
use libtt_metal_cxx::MeshDevice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtBufferLayout {
    Replicated,
    Sharded,
}

#[derive(Debug)]
struct TtBufferAllocation {
    mesh_buffer: MeshBuffer,
    layout: TtBufferLayout,
}

use crate::runtime::{TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES, TT_MEMORY_ALIGNMENT};

/// A GPU memory resource for TT-Metal.
#[derive(Debug, Clone)]
pub struct TtResource {
    pub storage_id: StorageId,
    pub owner_stream: StreamId,
    pub address: u32,
    pub size: u64,
    pub allocation_size: u64,
    pub allocation_offset: u64,
    pub compile_args: Vec<u32>,
    pub layout: TtBufferLayout,
}

// SAFETY: TtStorage is only accessed from one thread at a time.
unsafe impl Send for TtStorage {}

/// GPU storage for TT-Metal device memory.
///
/// Manages allocations using TT-Metal replicated `MeshBuffers` in DRAM.
/// All allocations are tile-aligned (multiples of `page_size`).
#[derive(Debug)]
pub struct TtStorage {
    mesh_ptr: *const MeshDevice,
    buffers: HashMap<StorageId, TtBufferAllocation>,
    next_id: StorageId,
    page_size: u64,
}

#[allow(clippy::new_without_default)]
impl TtStorage {
    pub fn new() -> Self {
        Self {
            mesh_ptr: std::ptr::null(),
            buffers: HashMap::new(),
            next_id: StorageId::new(),
            page_size: TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES,
        }
    }

    pub fn set_mesh_ptr(&mut self, ptr: *const MeshDevice) {
        self.mesh_ptr = ptr;
    }

    fn mesh(&self) -> &MeshDevice {
        assert!(!self.mesh_ptr.is_null(), "mesh_ptr not set on TtStorage");
        unsafe { &*self.mesh_ptr }
    }

    pub fn get_mesh_buffer(&self, id: StorageId) -> &MeshBuffer {
        &self
            .buffers
            .get(&id)
            .expect("MeshBuffer not found")
            .mesh_buffer
    }

    pub fn get_layout(&self, id: StorageId) -> TtBufferLayout {
        self.buffers.get(&id).expect("MeshBuffer not found").layout
    }
}

impl ComputeStorage for TtStorage {
    type Resource = TtResource;

    fn alignment(&self) -> usize {
        TT_MEMORY_ALIGNMENT as usize
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let id = handle.id;
        let buffer = self.buffers.get(&id).expect("Buffer not found");
        let compile_args = buffer
            .mesh_buffer
            .compile_args()
            .expect("failed to compute TT compile args for MeshBuffer");
        let address = u64::from(buffer.mesh_buffer.address())
            .checked_add(handle.offset())
            .and_then(|addr| u32::try_from(addr).ok())
            .expect("TT resource address should stay within 32-bit DRAM address space");
        TtResource {
            storage_id: id,
            owner_stream: StreamId { value: 0 },
            address,
            size: handle.size(),
            allocation_size: buffer.mesh_buffer.size(),
            allocation_offset: handle.offset(),
            compile_args,
            layout: buffer.layout,
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let mesh = self.mesh();
        let aligned_size = size.div_ceil(self.page_size) * self.page_size;
        let page_size = self.page_size;

        let buffer =
            MeshBuffer::create_replicated(mesh, aligned_size, page_size, 0).map_err(|e| {
                IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: format!("failed to allocate MeshBuffer: {}", e.what()),
                }
            })?;

        let id = self.next_id;
        self.next_id = StorageId::new();
        self.buffers.insert(
            id,
            TtBufferAllocation {
                mesh_buffer: buffer,
                layout: TtBufferLayout::Replicated,
            },
        );

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        self.buffers.remove(&id);
    }

    fn flush(&mut self) {
        // Nothing to flush in synchronous mode.
    }
}
