use cubecl_core::device::{Device, DeviceId};

#[derive(Clone, PartialEq, Eq, Default, Hash, Debug)]
pub struct MetaliumDevice {
    pub device_id: u16,
}

impl Device for MetaliumDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            device_id: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, self.device_id)
    }
}
