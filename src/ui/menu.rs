use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, Deserialize)]
pub struct MenuManifestTopLevel {
    #[serde(rename = "id")]
    pub _id: String,
    pub label: String,
    pub items: Vec<MenuManifestItem>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MenuManifestItem {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: Option<String>,
    pub label: Option<String>,
    pub command: Option<String>,
    pub shortcut: Option<String>,
    pub enabled: Option<bool>,
    pub items: Option<Vec<MenuManifestItem>>,
}

#[derive(Debug, Clone)]
pub struct MenuManifestCommand {
    pub id: String,
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
}

pub fn manifest() -> &'static Vec<MenuManifestTopLevel> {
    static MANIFEST: OnceLock<Vec<MenuManifestTopLevel>> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        serde_json::from_str(include_str!("menu/imagej-menu-manifest.json"))
            .expect("failed to parse menu manifest")
    })
}

pub fn manifest_commands() -> &'static Vec<MenuManifestCommand> {
    static COMMANDS: OnceLock<Vec<MenuManifestCommand>> = OnceLock::new();
    COMMANDS.get_or_init(|| {
        let mut entries = Vec::new();
        for top in manifest() {
            append_items(&top.items, &mut entries);
        }
        entries
    })
}

fn append_items(items: &[MenuManifestItem], output: &mut Vec<MenuManifestCommand>) {
    for item in items {
        match item.kind.as_str() {
            "item" => {
                let id = item
                    .command
                    .clone()
                    .or_else(|| item.id.clone())
                    .unwrap_or_default();
                if id.is_empty() {
                    continue;
                }

                output.push(MenuManifestCommand {
                    label: item.label.clone().unwrap_or_else(|| id.clone()),
                    id,
                    shortcut: item.shortcut.clone(),
                    enabled: item.enabled.unwrap_or(true),
                });
            }
            "submenu" => {
                if let Some(children) = item.items.as_deref() {
                    append_items(children, output);
                }
            }
            "separator" => {}
            _ => {
                if let Some(children) = item.items.as_deref() {
                    append_items(children, output);
                }
            }
        }
    }
}
