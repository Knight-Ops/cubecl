use cubecl_common::device::{Device, DeviceId};

/// A Tenstorrent device identified by its chip index.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TtDevice {
    pub index: usize,
}

impl Device for TtDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, self.index as u16)
    }
}
