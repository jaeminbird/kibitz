//! Local board vision: rectify the board, decide when the scene is settled,
//! track the board as the camera moves, and infer which legal move was played.
//! See docs/specs/cg-vision.md.

pub mod features;
pub mod gate;
pub mod geometry;
pub mod infer;
pub mod synth;
pub mod track;

pub use features::{square_diff, square_stats, SquareStats};
pub use gate::{GateConfig, GateState, StabilityGate};
pub use geometry::{grid_to_square, square_to_grid, warp_board, GrayImage, Homography, WarpedBoard};
pub use infer::{changed_squares, infer_move, InferConfig, Inference};
pub use track::{checker_score, CornerTracker, TrackerConfig};
