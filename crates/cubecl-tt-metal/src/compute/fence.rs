use cubecl_core::server::ServerError;

/// No-op fence for TT-Metal (synchronous execution).
///
/// All operations are blocking, so fences are unnecessary in the MVP.
#[derive(Debug)]
pub struct Fence;

impl Fence {
    pub fn new() -> Self {
        Self
    }

    pub async fn wait_sync(self) -> Result<(), ServerError> {
        Ok(())
    }
}

impl Default for Fence {
    fn default() -> Self {
        Self::new()
    }
}
