//! Error types used by `a3s-vec`.
//!
//! The public error intentionally mirrors zvec's status taxonomy while still
//! carrying a useful, typed Rust error.  Keeping the status code at the API
//! boundary makes it possible for adapters (CLI, HTTP, and A3S Code) to map
//! failures without parsing strings.

use std::fmt;

/// Stable status categories exposed by the collection API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    NotFound = 1,
    AlreadyExists = 2,
    InvalidArgument = 3,
    PermissionDenied = 4,
    FailedPrecondition = 5,
    ResourceExhausted = 6,
    Unavailable = 7,
    InternalError = 8,
    NotSupported = 9,
    Unknown = 10,
}

impl From<u32> for ErrorCode {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::NotFound,
            2 => Self::AlreadyExists,
            3 => Self::InvalidArgument,
            4 => Self::PermissionDenied,
            5 => Self::FailedPrecondition,
            6 => Self::ResourceExhausted,
            7 => Self::Unavailable,
            8 => Self::InternalError,
            9 => Self::NotSupported,
            _ => Self::Unknown,
        }
    }
}

impl From<ErrorCode> for u32 {
    fn from(value: ErrorCode) -> Self {
        value as u32
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::NotFound => "NotFound",
            Self::AlreadyExists => "AlreadyExists",
            Self::InvalidArgument => "InvalidArgument",
            Self::PermissionDenied => "PermissionDenied",
            Self::FailedPrecondition => "FailedPrecondition",
            Self::ResourceExhausted => "ResourceExhausted",
            Self::Unavailable => "Unavailable",
            Self::InternalError => "InternalError",
            Self::NotSupported => "NotSupported",
            Self::Unknown => "Unknown",
        };
        f.write_str(name)
    }
}

/// An error returned by a3s-vec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Stable machine-readable category.
    pub code: ErrorCode,
    /// Human-readable context.  Messages never contain a secret by design.
    pub message: String,
}

impl Error {
    /// Creates an error with a stable code and context.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn already_exists(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::AlreadyExists, message)
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::PermissionDenied, message)
    }

    pub fn failed_precondition(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::FailedPrecondition, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InternalError, message)
    }

    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceExhausted, message)
    }

    pub fn not_supported(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotSupported, message)
    }

    pub fn is_not_found(&self) -> bool {
        self.code == ErrorCode::NotFound
    }

    pub fn is_already_exists(&self) -> bool {
        self.code == ErrorCode::AlreadyExists
    }

    pub fn is_invalid_argument(&self) -> bool {
        self.code == ErrorCode::InvalidArgument
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a3s-vec error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Specialized result type for all public operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        let code = match value.kind() {
            std::io::ErrorKind::NotFound => ErrorCode::NotFound,
            std::io::ErrorKind::PermissionDenied => ErrorCode::PermissionDenied,
            std::io::ErrorKind::AlreadyExists => ErrorCode::AlreadyExists,
            _ => ErrorCode::InternalError,
        };
        Self::new(code, value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::new(ErrorCode::InternalError, format!("JSON error: {value}"))
    }
}

impl From<zvec_core::error::ZvecError> for Error {
    fn from(value: zvec_core::error::ZvecError) -> Self {
        use zvec_core::error::ZvecError;
        let (code, message) = match value {
            ZvecError::NotFound(message) => (ErrorCode::NotFound, message),
            ZvecError::AlreadyExists(message) => (ErrorCode::AlreadyExists, message),
            ZvecError::InvalidArgument(message) => (ErrorCode::InvalidArgument, message),
            ZvecError::PermissionDenied(message) => (ErrorCode::PermissionDenied, message),
            ZvecError::FailedPrecondition(message) => (ErrorCode::FailedPrecondition, message),
            ZvecError::ResourceExhausted(message) => (ErrorCode::ResourceExhausted, message),
            ZvecError::Unavailable(message) => (ErrorCode::Unavailable, message),
            ZvecError::Internal(message) => (ErrorCode::InternalError, message),
            ZvecError::NotSupported(message) => (ErrorCode::NotSupported, message),
            ZvecError::Unknown(message) => (ErrorCode::Unknown, message),
        };
        Self::new(code, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_are_stable() {
        assert_eq!(u32::from(ErrorCode::InvalidArgument), 3);
        assert_eq!(ErrorCode::from(99), ErrorCode::Unknown);
    }

    #[test]
    fn helpers_preserve_context() {
        let error = Error::not_found("document x");
        assert!(error.is_not_found());
        assert!(error.to_string().contains("document x"));
    }
}
