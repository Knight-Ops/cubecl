use cubecl_common::backtrace::BackTrace;
use cubecl_runtime::server::IoError;
use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::collections::HashMap;

use libtt_metal_cxx::MeshBuffer;
use libtt_metal_cxx::MeshDevice;

/// A GPU memory resource for TT-Metal.
#[derive(Debug, Clone)]
pub struct TtResource {
    pub storage_id: StorageId,
    pub address: u32,
    pub size: u64,
}

// SAFETY: TtStorage is only accessed from one thread at a time.
unsafe impl Send for TtStorage {}

/// GPU storage for TT-Metal device memory.
///
/// Manages allocations using TT-Metal replicated `MeshBuffers` in DRAM.
/// All allocations are tile-aligned (multiples of `32×32×element_size`).
#[derive(Debug)]
pub struct TtStorage {
    mesh_ptr: *const MeshDevice,
    buffers: HashMap<StorageId, MeshBuffer>,
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
            page_size: 2048, // default: one bfloat16 tile = 32*32*2 = 2048 bytes
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
        self.buffers.get(&id).expect("MeshBuffer not found")
    }
}

impl ComputeStorage for TtStorage {
    type Resource = TtResource;

    fn alignment(&self) -> usize {
        32 // Minimum TT-Metal alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let id = handle.id;
        let buffer = self.buffers.get(&id).expect("Buffer not found");
        TtResource {
            storage_id: id,
            address: buffer.address(),
            size: buffer.size(),
        }
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let mesh = self.mesh();
        // Tile-align: round up to nearest page_size boundary
        let aligned_size = size.div_ceil(self.page_size) * self.page_size;
        let page_size = self.page_size;

        let buffer = MeshBuffer::create_replicated(
            mesh,
            aligned_size,
            page_size,
            0, // DRAM
        )
        .map_err(|e| IoError::Unknown {
            backtrace: BackTrace::capture(),
            description: format!("failed to allocate MeshBuffer: {}", e.what()),
        })?;

        let id = self.next_id;
        self.next_id = StorageId::new();
        self.buffers.insert(id, buffer);

        Ok(StorageHandle::new(
            id,
            StorageUtilization {
                offset: 0,
                size: aligned_size,
            },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        // MeshBuffer is deallocated on drop
        self.buffers.remove(&id);
    }

    fn flush(&mut self) {
        // Nothing to flush in synchronous mode
    }
}
