use std::fs;
use std::path::Path;

use serde_json::json;
use tempfile::TempDir;

use super::{PLUGIN_API_VERSION, PluginCatalog, PluginCommandTarget};

const MINIMAL_COMPONENT: &[u8] = b"\0asm\x0d\0\x01\0";
const MINIMAL_CORE_MODULE: &[u8] = b"\0asm\x01\0\0\0";

const VALID_MANIFEST: &str = r#"
schema_version = 1
id = "org.example.contrast"
name = "Example Contrast"
version = "1.2.3"
description = "A catalog fixture."
authors = ["Image Scientist"]

[runtime]
kind = "wasm-component"
api_version = "0.1.0"
path = "extension.wasm"

[[operations]]
id = "stretch"
description = "Stretch image contrast."
export = "stretch"

[[operations.params]]
name = "minimum"
description = "Lower display bound."
required = false
kind = "float"

[[commands]]
id = "stretch"
label = "Stretch Contrast..."
menu_path = ["Plugins", "Examples"]
target = { kind = "operation", id = "stretch" }
default_params = { minimum = 2.5 }
"#;

const COMMAND_ONLY_MANIFEST: &str = r#"
schema_version = 1
id = "org.example.about"
name = "Example About"
version = "1.0.0"

[runtime]
kind = "wasm-component"
api_version = "0.1.0"
path = "extension.wasm"

[[handlers]]
id = "about"
description = "Show plugin information."
export = "show-about"

[[commands]]
id = "about"
label = "About Example..."
menu_path = ["Plugins", "Examples"]
target = { kind = "handler", id = "about" }
default_params = { argument = "credits" }
"#;

#[test]
fn discovers_valid_manifest_and_resolves_namespaced_contributions() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "contrast", VALID_MANIFEST);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert!(discovery.rejected.is_empty());
    let plugin = discovery
        .catalog
        .plugin("org.example.contrast")
        .expect("plugin");
    assert_eq!(plugin.name, "Example Contrast");
    assert_eq!(plugin.version.to_string(), "1.2.3");
    assert_eq!(plugin.api_version.to_string(), PLUGIN_API_VERSION);

    let operation = discovery
        .catalog
        .operation("org.example.contrast.stretch")
        .expect("operation");
    assert_eq!(operation.schema.name, operation.id);
    assert_eq!(operation.schema.params[0].name, "minimum");

    let command = discovery
        .catalog
        .command("org.example.contrast.stretch")
        .expect("command");
    assert_eq!(command.menu_path, ["Plugins", "Examples"]);
    assert_eq!(
        command.target,
        PluginCommandTarget::Operation {
            operation_id: operation.id.clone()
        }
    );
    assert_eq!(command.default_params["minimum"], json!(2.5));
}

#[test]
fn discovers_command_only_plugin_with_independent_handler() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "about", COMMAND_ONLY_MANIFEST);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert!(discovery.rejected.is_empty());
    assert_eq!(discovery.catalog.operations().len(), 0);
    let command = discovery
        .catalog
        .command("org.example.about.about")
        .expect("handler command");
    assert_eq!(
        command.target,
        PluginCommandTarget::Handler {
            handler_id: "org.example.about.about".to_string()
        }
    );
    assert_eq!(command.default_params["argument"], json!("credits"));
}

#[test]
fn missing_root_is_an_empty_catalog() {
    let root = TempDir::new().expect("root");
    let missing = root.path().join("not-created");
    let discovery = PluginCatalog::discover(missing).expect("missing root");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(discovery.rejected.is_empty());
}

#[test]
fn broken_package_is_isolated_from_valid_siblings() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "valid", VALID_MANIFEST);
    write_package(
        root.path(),
        "future",
        &VALID_MANIFEST.replace("schema_version = 1", "schema_version = 2"),
    );

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 1);
    assert_eq!(discovery.rejected.len(), 1);
    assert!(discovery.rejected[0].reason.contains("schema_version 2"));
}

#[test]
fn rejects_future_or_incompatible_runtime_api_versions() {
    for api_version in ["0.1.1", "0.2.0", "1.0.0"] {
        let root = TempDir::new().expect("root");
        let manifest = VALID_MANIFEST.replace(
            "api_version = \"0.1.0\"",
            &format!("api_version = \"{api_version}\""),
        );
        write_package(root.path(), "plugin", &manifest);
        let discovery = PluginCatalog::discover(root.path()).expect("discover");
        assert_eq!(discovery.catalog.plugins().len(), 0, "{api_version}");
        assert!(
            discovery.rejected[0]
                .reason
                .contains("incompatible plugin api_version"),
            "{api_version}"
        );
    }
}

#[test]
fn rejects_lexical_path_traversal() {
    let root = TempDir::new().expect("root");
    fs::write(root.path().join("escape.wasm"), MINIMAL_COMPONENT).expect("escape component");
    let manifest = VALID_MANIFEST.replace("path = \"extension.wasm\"", "path = \"../escape.wasm\"");
    write_package(root.path(), "plugin", &manifest);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(discovery.rejected[0].reason.contains("without traversal"));
}

#[cfg(unix)]
#[test]
fn rejects_component_symlink_that_escapes_the_package() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().expect("root");
    fs::write(root.path().join("escape.wasm"), MINIMAL_COMPONENT).expect("escape component");
    let package = root.path().join("plugin");
    fs::create_dir(&package).expect("package");
    fs::write(package.join(super::PLUGIN_MANIFEST_FILE), VALID_MANIFEST).expect("manifest");
    symlink(
        root.path().join("escape.wasm"),
        package.join("extension.wasm"),
    )
    .expect("symlink");

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("outside its plugin package")
    );
}

#[cfg(unix)]
#[test]
fn rejects_symlinked_manifests() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().expect("root");
    let external_manifest = root.path().join("external.toml");
    fs::write(&external_manifest, VALID_MANIFEST).expect("external manifest");
    let package = root.path().join("plugin");
    fs::create_dir(&package).expect("package");
    fs::write(package.join("extension.wasm"), MINIMAL_COMPONENT).expect("component");
    symlink(
        &external_manifest,
        package.join(super::PLUGIN_MANIFEST_FILE),
    )
    .expect("manifest symlink");

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("regular file rather than a symlink")
    );
}

#[test]
fn duplicate_plugin_ids_have_deterministic_first_package_ownership() {
    let root = TempDir::new().expect("root");
    write_package(
        root.path(),
        "a-first",
        &VALID_MANIFEST.replace("Example Contrast", "First Package"),
    );
    write_package(
        root.path(),
        "z-last",
        &VALID_MANIFEST.replace("Example Contrast", "Last Package"),
    );

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(
        discovery
            .catalog
            .plugin("org.example.contrast")
            .unwrap()
            .name,
        "First Package"
    );
    assert_eq!(discovery.rejected.len(), 1);
    assert!(discovery.rejected[0].reason.contains("duplicate plugin id"));
}

#[test]
fn rejects_duplicate_local_ids_and_unresolved_command_targets() {
    let root = TempDir::new().expect("root");
    let duplicate = format!(
        "{VALID_MANIFEST}\n[[operations]]\nid = \"stretch\"\ndescription = \"Again\"\nexport = \"again\"\n"
    );
    write_package(root.path(), "duplicate", &duplicate);

    let missing_target = VALID_MANIFEST.replace(
        "target = { kind = \"operation\", id = \"stretch\" }",
        "target = { kind = \"operation\", id = \"missing\" }",
    );
    write_package(root.path(), "missing", &missing_target);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    let reasons = discovery
        .rejected
        .iter()
        .map(|rejection| rejection.reason.as_str())
        .collect::<Vec<_>>();
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("duplicate local operation id"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("undeclared local operation"))
    );
}

#[test]
fn rejects_dotted_local_ids_to_keep_qualification_injective() {
    let root = TempDir::new().expect("root");
    let dotted = VALID_MANIFEST.replacen("id = \"stretch\"", "id = \"group.stretch\"", 1);
    write_package(root.path(), "dotted", &dotted);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("one lowercase identifier without dots")
    );
}

#[test]
fn rejects_core_menu_injection_and_unknown_default_parameters() {
    let root = TempDir::new().expect("root");
    let core_menu = VALID_MANIFEST.replace(
        "menu_path = [\"Plugins\", \"Examples\"]",
        "menu_path = [\"File\", \"Import\"]",
    );
    write_package(root.path(), "core-menu", &core_menu);

    let unknown_param = VALID_MANIFEST.replace(
        "default_params = { minimum = 2.5 }",
        "default_params = { maximum = 2.5 }",
    );
    write_package(root.path(), "unknown-param", &unknown_param);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    let reasons = discovery
        .rejected
        .iter()
        .map(|rejection| rejection.reason.as_str())
        .collect::<Vec<_>>();
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("must start with `Plugins`"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("undeclared parameter"))
    );
}

#[test]
fn rejects_unknown_parameter_kinds_and_mismatched_defaults() {
    let root = TempDir::new().expect("root");
    let unknown_kind = VALID_MANIFEST.replace("kind = \"float\"", "kind = \"number\"");
    write_package(root.path(), "unknown-kind", &unknown_kind);

    let mismatched = VALID_MANIFEST.replace(
        "default_params = { minimum = 2.5 }",
        "default_params = { minimum = \"low\" }",
    );
    write_package(root.path(), "mismatched", &mismatched);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    let reasons = discovery
        .rejected
        .iter()
        .map(|rejection| rejection.reason.as_str())
        .collect::<Vec<_>>();
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("unsupported parameter kind"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("does not match declared kind"))
    );
}

#[test]
fn accepts_the_core_parameter_vocabulary_and_checks_integer_arrays() {
    let root = TempDir::new().expect("root");
    let extra_params = r#"
[[operations.params]]
name = "flag"
description = "Boolean."
required = false
kind = "bool"

[[operations.params]]
name = "count"
description = "Integer."
required = false
kind = "integer"

[[operations.params]]
name = "label"
description = "String."
required = false
kind = "string"

[[operations.params]]
name = "values"
description = "Array."
required = false
kind = "array"

[[operations.params]]
name = "indices"
description = "Integer array."
required = false
kind = "array<int>"

[[operations.params]]
name = "config"
description = "Object."
required = false
kind = "object"
"#;
    let manifest = VALID_MANIFEST
        .replace("\n[[commands]]", &format!("\n{extra_params}\n[[commands]]"))
        .replace(
            "default_params = { minimum = 2.5 }",
            "default_params = { minimum = 2, flag = true, count = 3, label = \"ok\", values = [1, 2], indices = [1, 2], config = { mode = \"safe\" } }",
        );
    write_package(root.path(), "valid-kinds", &manifest);

    let invalid_array = manifest
        .replace(
            "id = \"org.example.contrast\"",
            "id = \"org.example.bad-array\"",
        )
        .replace("indices = [1, 2]", "indices = [1.0, 2.0]");
    write_package(root.path(), "bad-array", &invalid_array);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 1);
    assert_eq!(discovery.rejected.len(), 1);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("does not match declared kind")
    );
}

#[test]
fn bounds_authors_contributions_and_operation_parameters() {
    let root = TempDir::new().expect("root");

    let authors = (0..33)
        .map(|index| format!("\"Author {index}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let too_many_authors = VALID_MANIFEST.replace(
        "authors = [\"Image Scientist\"]",
        &format!("authors = [{authors}]"),
    );
    write_package(root.path(), "authors", &too_many_authors);

    let mut contribution_body = String::new();
    for index in 0..128 {
        contribution_body.push_str(&format!(
            "\n[[operations]]\nid = \"op{index}\"\ndescription = \"Operation {index}.\"\nexport = \"op{index}\"\n"
        ));
    }
    for index in 0..128 {
        contribution_body.push_str(&format!(
            "\n[[handlers]]\nid = \"handler{index}\"\ndescription = \"Handler {index}.\"\nexport = \"handler{index}\"\n"
        ));
    }
    contribution_body.push_str(
        "\n[[commands]]\nid = \"run\"\nlabel = \"Run\"\nmenu_path = [\"Plugins\"]\ntarget = { kind = \"operation\", id = \"op0\" }\n",
    );
    write_package(
        root.path(),
        "contributions",
        &manifest_with_body("org.example.contributions", &contribution_body),
    );

    let mut parameter_body =
        "\n[[operations]]\nid = \"bounded\"\ndescription = \"Bounded.\"\nexport = \"bounded\"\n"
            .to_string();
    for index in 0..65 {
        parameter_body.push_str(&format!(
            "\n[[operations.params]]\nname = \"p{index}\"\ndescription = \"Parameter.\"\nrequired = false\nkind = \"int\"\n"
        ));
    }
    write_package(
        root.path(),
        "parameters",
        &manifest_with_body("org.example.parameters", &parameter_body),
    );

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    let reasons = discovery
        .rejected
        .iter()
        .map(|rejection| rejection.reason.as_str())
        .collect::<Vec<_>>();
    assert!(reasons.iter().any(|reason| reason.contains("33 authors")));
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("257 total contributions"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("65 parameters for operation"))
    );
}

#[test]
fn bounds_manifest_and_default_value_resources() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "manifest-size", &"x".repeat(256 * 1024 + 1));

    let nested = format!("{}1{}", "[".repeat(9), "]".repeat(9));
    let deep_defaults = COMMAND_ONLY_MANIFEST.replace(
        "default_params = { argument = \"credits\" }",
        &format!("default_params = {{ payload = {nested} }}"),
    );
    write_package(root.path(), "deep-defaults", &deep_defaults);

    let values = std::iter::repeat_n("1", 1024).collect::<Vec<_>>().join(",");
    let many_defaults = COMMAND_ONLY_MANIFEST.replace(
        "default_params = { argument = \"credits\" }",
        &format!("default_params = {{ payload = [{values}] }}"),
    );
    write_package(root.path(), "many-defaults", &many_defaults);

    let large_defaults = COMMAND_ONLY_MANIFEST.replace(
        "default_params = { argument = \"credits\" }",
        &format!(
            "default_params = {{ payload = \"{}\" }}",
            "x".repeat(64 * 1024)
        ),
    );
    write_package(root.path(), "large-defaults", &large_defaults);

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    let reasons = discovery
        .rejected
        .iter()
        .map(|rejection| rejection.reason.as_str())
        .collect::<Vec<_>>();
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("262144-byte limit"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("nesting depth"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("default-value nodes"))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| reason.contains("65536-byte default-value limit"))
    );
}

#[test]
fn rejects_non_wasm_runtime_artifacts() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "plugin", VALID_MANIFEST);
    fs::write(root.path().join("plugin/extension.wasm"), b"nope").expect("bad component");

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("must use WebAssembly Component encoding")
    );
}

#[test]
fn rejects_core_wasm_modules_when_a_component_is_required() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "plugin", VALID_MANIFEST);
    fs::write(
        root.path().join("plugin/extension.wasm"),
        MINIMAL_CORE_MODULE,
    )
    .expect("core module");

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("core modules are not accepted")
    );
}

#[test]
fn rejects_malformed_component_bodies_after_a_valid_header() {
    let root = TempDir::new().expect("root");
    write_package(root.path(), "plugin", VALID_MANIFEST);
    let mut malformed = MINIMAL_COMPONENT.to_vec();
    malformed.push(0xff);
    fs::write(root.path().join("plugin/extension.wasm"), malformed).expect("malformed component");

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(
        discovery.rejected[0]
            .reason
            .contains("is not a valid WebAssembly Component")
    );
}

#[test]
fn ignores_nested_packages_and_symlinked_package_directories() {
    let root = TempDir::new().expect("root");
    let nested_parent = root.path().join("nested");
    fs::create_dir(&nested_parent).expect("nested parent");
    write_package(&nested_parent, "plugin", VALID_MANIFEST);

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        symlink(
            nested_parent.join("plugin"),
            root.path().join("linked-plugin"),
        )
        .expect("symlink package");
    }

    let discovery = PluginCatalog::discover(root.path()).expect("discover");
    assert_eq!(discovery.catalog.plugins().len(), 0);
    assert!(discovery.rejected.is_empty());
}

fn write_package(root: &Path, directory: &str, manifest: &str) {
    let package = root.join(directory);
    fs::create_dir(&package).expect("package directory");
    fs::write(package.join(super::PLUGIN_MANIFEST_FILE), manifest).expect("manifest");
    fs::write(package.join("extension.wasm"), MINIMAL_COMPONENT).expect("component");
}

fn manifest_with_body(plugin_id: &str, body: &str) -> String {
    format!(
        r#"schema_version = 1
id = "{plugin_id}"
name = "Generated Fixture"
version = "1.0.0"

[runtime]
kind = "wasm-component"
api_version = "0.1.0"
path = "extension.wasm"
{body}
"#
    )
}
