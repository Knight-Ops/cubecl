use cubecl_runtime::storage::{ComputeStorage, StorageHandle, StorageId};
use cubecl_runtime::server::IoError;
use std::collections::HashMap;

/// A GPU memory resource for TT-Metal (wraps a MeshBuffer).
#[derive(Debug)]
pub struct TtResource {
    pub address: u32,
    pub size: u64,
}

// SAFETY: TtStorage is only accessed from one thread at a time.
unsafe impl Send for TtStorage {}

/// GPU storage for TT-Metal device memory.
///
/// Manages allocations using TT-Metal interleaved DRAM buffers.
/// All allocations must be tile-aligned (multiples of 32×32×element_size).
#[derive(Debug)]
pub struct TtStorage {
    device: Option<libtt_metal_cxx::Device>,
    mesh_device: Option<libtt_metal_cxx::MeshDevice>,
    buffers: HashMap<StorageId, libtt_metal_cxx::Buffer>,
    next_id: StorageId,
}

impl TtStorage {
    pub fn new() -> Self {
        Self {
            device: None,
            mesh_device: None,
            buffers: HashMap::new(),
            next_id: StorageId::new(),
        }
    }
}

impl ComputeStorage for TtStorage {
    type Resource = TtResource;

    fn alignment(&self) -> usize {
        32 // Minimum TT-Metal alignment
    }

    fn get(&mut self, _handle: &StorageHandle) -> Self::Resource {
        todo!("TT-Metal storage get not yet implemented")
    }

    fn alloc(&mut self, _size: u64) -> Result<StorageHandle, IoError> {
        todo!("TT-Metal storage alloc not yet implemented")
    }

    fn dealloc(&mut self, id: StorageId) {
        if let Some(mut buffer) = self.buffers.remove(&id) {
            let _ = buffer.deallocate();
        }
    }

    fn flush(&mut self) {
        // Nothing to flush in synchronous mode
    }
}
