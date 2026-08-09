use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// ImageJ-compatible tool identifiers. The labels are deliberately short because
/// they are also used by macros and the compact status line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) enum ToolId {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
    Angle,
    Point,
    Wand,
    Text,
    Zoom,
    Hand,
    Dropper,
    More,
}

impl ToolId {
    pub(super) const fn command_id(self) -> &'static str {
        match self {
            Self::Rect => "launcher.tool.rect",
            Self::Oval => "launcher.tool.oval",
            Self::Poly => "launcher.tool.poly",
            Self::Free => "launcher.tool.free",
            Self::Line => "launcher.tool.line",
            Self::Angle => "launcher.tool.angle",
            Self::Point => "launcher.tool.point",
            Self::Wand => "launcher.tool.wand",
            Self::Text => "launcher.tool.text",
            Self::Zoom => "launcher.tool.zoom",
            Self::Hand => "launcher.tool.hand",
            Self::Dropper => "launcher.tool.dropper",
            Self::More => "launcher.tool.more",
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Rect => "Rectangle",
            Self::Oval => "Oval",
            Self::Poly => "Polygon",
            Self::Free => "Freehand",
            Self::Line => "Line",
            Self::Angle => "Angle",
            Self::Point => "Point",
            Self::Wand => "Wand",
            Self::Text => "Text",
            Self::Zoom => "Magnifier",
            Self::Hand => "Scrolling Tool",
            Self::Dropper => "Color Picker",
            Self::More => "More Tools",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ToolbarItem {
    pub(super) tool: ToolId,
    pub(super) icon: &'static str,
}

pub(super) const TOOLBAR_ITEMS: &[ToolbarItem] = &[
    ToolbarItem {
        tool: ToolId::Rect,
        icon: "rect.svg",
    },
    ToolbarItem {
        tool: ToolId::Oval,
        icon: "oval.svg",
    },
    ToolbarItem {
        tool: ToolId::Poly,
        icon: "poly.svg",
    },
    ToolbarItem {
        tool: ToolId::Free,
        icon: "free.svg",
    },
    ToolbarItem {
        tool: ToolId::Line,
        icon: "line.svg",
    },
    ToolbarItem {
        tool: ToolId::Angle,
        icon: "angle.svg",
    },
    ToolbarItem {
        tool: ToolId::Point,
        icon: "point.svg",
    },
    ToolbarItem {
        tool: ToolId::Wand,
        icon: "wand.svg",
    },
    ToolbarItem {
        tool: ToolId::Text,
        icon: "text.svg",
    },
    ToolbarItem {
        tool: ToolId::Zoom,
        icon: "zoom.svg",
    },
    ToolbarItem {
        tool: ToolId::Hand,
        icon: "hand.svg",
    },
    ToolbarItem {
        tool: ToolId::Dropper,
        icon: "dropper.svg",
    },
    ToolbarItem {
        tool: ToolId::More,
        icon: "more.svg",
    },
];

pub(super) fn icon_path(icon: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("ui")
        .join("assets")
        .join("lucide")
        .join(icon)
}

pub(super) fn tool_from_shortcut(key: &str) -> Option<ToolId> {
    match key {
        "r" => Some(ToolId::Rect),
        "o" => Some(ToolId::Oval),
        "g" => Some(ToolId::Poly),
        "f" => Some(ToolId::Free),
        "l" => Some(ToolId::Line),
        "a" => Some(ToolId::Angle),
        "p" | "." => Some(ToolId::Point),
        "w" => Some(ToolId::Wand),
        "t" => Some(ToolId::Text),
        "z" => Some(ToolId::Zoom),
        "h" => Some(ToolId::Hand),
        "d" => Some(ToolId::Dropper),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imagej_tool_shortcuts_are_preserved() {
        assert_eq!(tool_from_shortcut("r"), Some(ToolId::Rect));
        assert_eq!(tool_from_shortcut("z"), Some(ToolId::Zoom));
        assert_eq!(tool_from_shortcut("d"), Some(ToolId::Dropper));
    }

    #[test]
    fn internal_tool_ids_remain_macro_compatible() {
        assert_eq!(ToolId::Rect.command_id(), "launcher.tool.rect");
        assert_eq!(ToolId::More.command_id(), "launcher.tool.more");
    }
}
