use super::storage::gpu::{TtResource, TtStorage};
use crate::{
    compute::{command::Command, context::TtContext, stream::TtStreamBackend},
    runtime::TtCompiler,
};
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration, backtrace::BackTrace,
    ir::MemoryDeviceProperties, prelude::*,
    server::{
        Binding, CopyDescriptor, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::ManagedResource,
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct TtServer {
    ctx: TtContext,
    streams: MultiStream<TtStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

// SAFETY: `TtServer` is only accessed from one thread at a time via the `DeviceHandle`.
unsafe impl Send for TtServer {}

impl ComputeServer for TtServer {
    type Kernel = Box<dyn CubeTask<TtCompiler>>;
    type Storage = TtStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        sizes: &[usize],
        stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        Ok(sizes
            .iter()
            .map(|size| command.reserve(*size as u64).map(|_| Bytes::from_bytes_vec(vec![0u8; *size])))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_default())
    }

    fn initialize_memory(&mut self, _memory: ManagedMemoryHandle, _size: u64, _stream_id: StreamId) {
        // TODO: implement TT-Metal memory initialization
    }

    fn read(
        &mut self,
        _descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        Box::pin(async { Err(ServerError::ServerUnhealthy {
            errors: vec![ServerError::Generic {
                reason: "read not yet implemented for TT-Metal".into(),
                backtrace: BackTrace::capture(),
            }],
            backtrace: BackTrace::capture(),
        })})
    }

    fn write(&mut self, _descriptors: Vec<(CopyDescriptor, Bytes)>, _stream_id: StreamId) {
        // TODO: implement TT-Metal write
    }

    unsafe fn launch(
        &mut self,
        _kernel: Self::Kernel,
        _count: CubeCount,
        _bindings: KernelArguments,
        _mode: ExecutionMode,
        _stream_id: StreamId,
    ) {
        // TODO: implement TT-Metal kernel launch
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        Box::pin(async { Ok(()) })
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<TtResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command.resource(binding)
            .map_err(|e| ServerError::Generic {
                reason: format!("{e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        Ok(MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        })
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {}

    fn allocation_mode(&mut self, _mode: MemoryAllocationMode, _stream_id: StreamId) {}
}

impl ServerCommunication for TtServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl TtServer {
    pub(crate) fn new(
        ctx: TtContext,
        _mem_props: MemoryDeviceProperties,
        _mem_config: MemoryConfiguration,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            ctx,
            streams: MultiStream::new(
                utilities.logger.clone(),
                TtStreamBackend::new(),
                max_streams,
            ),
            utilities: Arc::new(utilities),
        }
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;
        Ok(Command::new(&mut self.ctx, streams))
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
