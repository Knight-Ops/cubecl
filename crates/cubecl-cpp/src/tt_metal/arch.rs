use std::fmt::Display;

use crate::shared::Architecture;

/// TT-Metal hardware architecture.
///
/// Tensix cores operate on 32×32 tiles natively.
/// The warp size (32) reflects the row width of a single tile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TtArchitecture {
    #[default]
    Wormhole,
    Blackhole,
    Other,
}

impl Display for TtArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Wormhole => write!(f, "wormhole"),
            Self::Blackhole => write!(f, "blackhole"),
            Self::Other => write!(f, "other"),
        }
    }
}

impl TtArchitecture {
    pub fn parse(arg: &str) -> Result<Self, String> {
        let norm = arg.to_lowercase();
        if norm.starts_with("wormhole") {
            Ok(TtArchitecture::Wormhole)
        } else if norm.starts_with("blackhole") {
            Ok(TtArchitecture::Blackhole)
        } else {
            Ok(TtArchitecture::Other)
        }
    }
}

impl Architecture for TtArchitecture {
    fn warp_size(&self) -> u32 {
        32 // Tensix tile row width
    }

    fn is_wmma_capable(&self) -> bool {
        false // No tensor cores; uses FPU for matrix operations
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }
}
