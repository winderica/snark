//! Core interface for working with Rank-1 Constraint Systems (R1CS).

use ark_std::vec::Vec;

/// A result type specialized to `SynthesisError`.
pub type Result<T> = core::result::Result<T, SynthesisError>;

#[macro_use]
mod impl_lc;
mod constraint_system;
mod error;
#[cfg(feature = "tracing")]
mod trace;

#[cfg(feature = "tracing")]
pub use crate::r1cs::trace::{ConstraintLayer, ConstraintTrace, TraceStep, TracingMode};

pub use tracing::info_span;

pub use ark_ff::{Field, ToConstraintField};
pub use constraint_system::{
    ConstraintMatrices, ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, Namespace,
    OptimizationGoal, SynthesisMode,
};
pub use error::SynthesisError;

/// A sparse representation of constraint matrices.
pub type Matrix<F> = Vec<Vec<(F, usize)>>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
/// An opaque counter for symbolic linear combinations.
pub struct LcIndex(pub usize);

/// Represents the different kinds of variables present in a constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Variable {
    /// Represents the "zero" constant.
    Zero,
    /// Represents of the "one" constant.
    One,
    /// Represents a public instance variable.
    Instance(usize),
    /// Represents a private witness variable.
    Committed(usize),
    Witness(usize),
    /// Represents of a linear combination.
    SymbolicLc(LcIndex),
}

/// A linear combination of variables according to associated coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LinearCombination<F: Field>(pub Vec<(F, Variable)>);

/// Generate a `Namespace` with name `name` from `ConstraintSystem` `cs`.
/// `name` must be a `&'static str`.
#[macro_export]
macro_rules! ns {
    ($cs:expr, $name:expr) => {{
        let span = $crate::r1cs::info_span!(target: "r1cs", $name);
        let id = span.id();
        let _enter_guard = span.enter();
        core::mem::forget(_enter_guard);
        core::mem::forget(span);
        $crate::r1cs::Namespace::new($cs.clone(), id)
    }};
}

impl Variable {
    /// Is `self` the zero variable?
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, Variable::Zero)
    }

    /// Is `self` the one variable?
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self, Variable::One)
    }

    /// Is `self` an instance variable?
    #[inline]
    pub fn is_instance(&self) -> bool {
        matches!(self, Variable::Instance(_))
    }

    /// Is `self` a witness variable?
    #[inline]
    pub fn is_witness(&self) -> bool {
        matches!(self, Variable::Witness(_))
    }

    /// Is `self` a linear combination?
    #[inline]
    pub fn is_lc(&self) -> bool {
        matches!(self, Variable::SymbolicLc(_))
    }

    /// Get the `LcIndex` in `self` if `self.is_lc()`.
    #[inline]
    pub fn get_lc_index(&self) -> Option<LcIndex> {
        match self {
            Variable::SymbolicLc(index) => Some(*index),
            _ => None,
        }
    }
}