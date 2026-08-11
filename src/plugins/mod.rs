//! Safe discovery and validation for external image-rs plugins.
//!
//! The versioned WIT contract and host-side buffer/progress invariants live in [`contract`]. This
//! milestone still stops before executing plugin code: a catalog contains only validated metadata
//! and declarative contributions, and a future WebAssembly Component adapter can remain behind
//! this module without exposing loader details to callers.

pub mod contract;
mod manifest;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use semver::Version;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::commands::OpSchema;

use manifest::ValidatedPackage;

/// File name recognized inside each immediate child of a plugin root.
pub const PLUGIN_MANIFEST_FILE: &str = "image-rs-plugin.toml";

/// The only manifest schema understood by this release.
pub const PLUGIN_SCHEMA_VERSION: u32 = 1;

/// Host interface version implemented by the bundled WIT contract.
pub const PLUGIN_API_VERSION: &str = "0.1.0";

/// A discovered plugin's stable metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginDescriptor {
    pub id: String,
    pub name: String,
    pub version: Version,
    pub description: Option<String>,
    pub authors: Vec<String>,
    pub api_version: Version,
}

/// An operation declared by a plugin.
///
/// `id` is always namespaced as `<plugin-id>.<local-id>`. The executable target remains private
/// to this module so callers cannot bypass the eventual sandbox adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginOperationContribution {
    pub id: String,
    pub plugin_id: String,
    pub schema: OpSchema,
    export: String,
}

/// A command declared by a plugin and resolved to an operation or independent handler.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginCommandContribution {
    pub id: String,
    pub plugin_id: String,
    pub label: String,
    pub menu_path: Vec<String>,
    pub target: PluginCommandTarget,
    pub default_params: Map<String, Value>,
}

/// The host-visible kind and stable identity of a command's executable target.
///
/// Handler export names remain private to the plugin module. Keeping the target tagged also
/// allows an operation and handler to use the same local identifier without ambiguity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginCommandTarget {
    Operation { operation_id: String },
    Handler { handler_id: String },
}

/// Deterministic, validated plugin metadata and contributions.
///
/// Discovery never loads or executes code. Iterators are ordered lexicographically by fully
/// qualified identifier.
#[derive(Debug, Clone, Default)]
pub struct PluginCatalog {
    plugins: BTreeMap<String, PluginDescriptor>,
    operations: BTreeMap<String, PluginOperationContribution>,
    commands: BTreeMap<String, PluginCommandContribution>,
    packages: BTreeMap<String, PackageRuntime>,
}

#[derive(Debug, Clone)]
struct PackageRuntime {
    /// Intentionally private until the sandboxed execution adapter exists.
    #[allow(dead_code)]
    component_path: PathBuf,
    /// Handler exports are intentionally inaccessible until sandboxed invocation exists.
    #[allow(dead_code)]
    handler_exports: BTreeMap<String, String>,
}

/// Discovery succeeded for the root, potentially with independently rejected packages.
#[derive(Debug, Clone, Default)]
pub struct PluginDiscovery {
    pub catalog: PluginCatalog,
    pub rejected: Vec<RejectedPlugin>,
}

/// A package containing a manifest was ignored because it was invalid or conflicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedPlugin {
    pub package_path: PathBuf,
    pub reason: String,
}

#[derive(Debug, Error)]
pub enum PluginDiscoveryError {
    #[error("failed to inspect plugin root `{path}`: {source}")]
    InspectRoot {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl PluginCatalog {
    /// Discover packages in immediate child directories of `root`.
    ///
    /// A missing root is an empty catalog, which keeps first-run application startup simple.
    /// Other root I/O failures abort discovery. Once the root is readable, a broken package is
    /// isolated in `PluginDiscovery::rejected` and does not suppress valid siblings.
    pub fn discover(root: impl AsRef<Path>) -> Result<PluginDiscovery, PluginDiscoveryError> {
        let root = root.as_ref();
        let entries = match fs::read_dir(root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(PluginDiscovery::default());
            }
            Err(source) => {
                return Err(PluginDiscoveryError::InspectRoot {
                    path: root.to_path_buf(),
                    source,
                });
            }
        };

        let mut package_paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| PluginDiscoveryError::InspectRoot {
                path: root.to_path_buf(),
                source,
            })?;
            let file_type =
                entry
                    .file_type()
                    .map_err(|source| PluginDiscoveryError::InspectRoot {
                        path: root.to_path_buf(),
                        source,
                    })?;
            // Do not follow a symlinked package directory outside the configured plugin root.
            if file_type.is_dir() && entry.path().join(PLUGIN_MANIFEST_FILE).is_file() {
                package_paths.push(entry.path());
            }
        }
        package_paths.sort();

        let mut discovery = PluginDiscovery::default();
        for package_path in package_paths {
            match manifest::load_package(&package_path) {
                Ok(package) => {
                    if let Err(reason) = discovery.catalog.install(package) {
                        discovery.rejected.push(RejectedPlugin {
                            package_path,
                            reason,
                        });
                    }
                }
                Err(reason) => discovery.rejected.push(RejectedPlugin {
                    package_path,
                    reason,
                }),
            }
        }
        Ok(discovery)
    }

    pub fn plugins(&self) -> impl ExactSizeIterator<Item = &PluginDescriptor> {
        self.plugins.values()
    }

    pub fn plugin(&self, id: &str) -> Option<&PluginDescriptor> {
        self.plugins.get(id)
    }

    pub fn operations(&self) -> impl ExactSizeIterator<Item = &PluginOperationContribution> {
        self.operations.values()
    }

    pub fn operation(&self, id: &str) -> Option<&PluginOperationContribution> {
        self.operations.get(id)
    }

    pub fn commands(&self) -> impl ExactSizeIterator<Item = &PluginCommandContribution> {
        self.commands.values()
    }

    pub fn command(&self, id: &str) -> Option<&PluginCommandContribution> {
        self.commands.get(id)
    }

    fn install(&mut self, package: ValidatedPackage) -> Result<(), String> {
        let ValidatedPackage {
            descriptor,
            component_path,
            handler_exports,
            operations,
            commands,
        } = package;
        let plugin_id = descriptor.id.clone();
        if self.plugins.contains_key(&plugin_id) {
            return Err(format!(
                "duplicate plugin id `{plugin_id}`; the lexicographically earlier package owns it"
            ));
        }

        if let Some(conflict) = operations
            .keys()
            .find(|id| self.operations.contains_key(*id))
        {
            return Err(format!("duplicate operation contribution id `{conflict}`"));
        }
        if let Some(conflict) = commands.keys().find(|id| self.commands.contains_key(*id)) {
            return Err(format!("duplicate command contribution id `{conflict}`"));
        }

        self.packages.insert(
            plugin_id.clone(),
            PackageRuntime {
                component_path,
                handler_exports,
            },
        );
        self.operations.extend(operations);
        self.commands.extend(commands);
        self.plugins.insert(plugin_id, descriptor);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
