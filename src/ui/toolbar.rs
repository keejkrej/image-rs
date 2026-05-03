use super::interaction;
use super::interaction::tooling::ToolId;

#[derive(Debug, Clone, Copy)]
pub(super) struct LauncherToolbarItem {
    pub(super) kind: ToolbarKind,
    pub(super) command_id: &'static str,
    pub(super) label: &'static str,
    pub(super) icon: ToolbarIcon,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ViewerToolbarItem {
    pub(super) kind: ToolbarKind,
    pub(super) command_id: &'static str,
    pub(super) label: &'static str,
    pub(super) glyph: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolbarKind {
    Button,
    Separator,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum ToolbarIcon {
    Rect,
    Oval,
    Poly,
    Free,
    Line,
    Angle,
    Point,
    Wand,
    Text,
    Open,
    Zoom,
    Hand,
    Dropper,
    Previous,
    Next,
    Quit,
    Custom1,
    Custom2,
    Custom3,
    More,
}

pub(super) const fn launcher_toolbar_button(
    command_id: &'static str,
    label: &'static str,
    icon: ToolbarIcon,
) -> LauncherToolbarItem {
    LauncherToolbarItem {
        kind: ToolbarKind::Button,
        command_id,
        label,
        icon,
    }
}

pub(super) const fn launcher_toolbar_separator() -> LauncherToolbarItem {
    LauncherToolbarItem {
        kind: ToolbarKind::Separator,
        command_id: "",
        label: "",
        icon: ToolbarIcon::Rect,
    }
}

pub(super) const fn viewer_toolbar_button(
    command_id: &'static str,
    label: &'static str,
    glyph: &'static str,
) -> ViewerToolbarItem {
    ViewerToolbarItem {
        kind: ToolbarKind::Button,
        command_id,
        label,
        glyph,
    }
}

pub(super) const fn viewer_toolbar_separator() -> ViewerToolbarItem {
    ViewerToolbarItem {
        kind: ToolbarKind::Separator,
        command_id: "",
        label: "",
        glyph: "",
    }
}

pub(super) fn launcher_toolbar_items() -> &'static [LauncherToolbarItem] {
    const ITEMS: &[LauncherToolbarItem] = &[
        launcher_toolbar_button("launcher.tool.rect", "Rectangle", ToolbarIcon::Rect),
        launcher_toolbar_button("launcher.tool.oval", "Oval", ToolbarIcon::Oval),
        launcher_toolbar_button("launcher.tool.poly", "Polygon", ToolbarIcon::Poly),
        launcher_toolbar_button("launcher.tool.free", "Freehand", ToolbarIcon::Free),
        launcher_toolbar_button("launcher.tool.line", "Line", ToolbarIcon::Line),
        launcher_toolbar_button("launcher.tool.angle", "Angle", ToolbarIcon::Angle),
        launcher_toolbar_button("launcher.tool.point", "Point", ToolbarIcon::Point),
        launcher_toolbar_button("launcher.tool.wand", "Wand", ToolbarIcon::Wand),
        launcher_toolbar_button("launcher.tool.text", "Text", ToolbarIcon::Text),
        launcher_toolbar_separator(),
        launcher_toolbar_button("launcher.tool.zoom", "Zoom", ToolbarIcon::Zoom),
        launcher_toolbar_button("launcher.tool.hand", "Hand", ToolbarIcon::Hand),
        launcher_toolbar_button("launcher.tool.dropper", "Dropper", ToolbarIcon::Dropper),
        launcher_toolbar_separator(),
        launcher_toolbar_button("launcher.tool.more", "More Tools", ToolbarIcon::More),
    ];
    ITEMS
}

pub(super) fn viewer_toolbar_items() -> &'static [ViewerToolbarItem] {
    const ITEMS: &[ViewerToolbarItem] = &[
        viewer_toolbar_button("file.open", "Open", "O"),
        viewer_toolbar_button("file.close", "Close", "C"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("image.zoom.in", "Zoom In", "+"),
        viewer_toolbar_button("image.zoom.out", "Zoom Out", "-"),
        viewer_toolbar_button("image.zoom.original", "Original Scale", "Orig"),
        viewer_toolbar_button("image.zoom.view100", "View 100%", "100%"),
        viewer_toolbar_button("image.zoom.to_selection", "To Selection", "Sel"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("process.smooth", "Smooth", "S"),
        viewer_toolbar_button("process.gaussian", "Gaussian Blur", "G"),
        viewer_toolbar_separator(),
        viewer_toolbar_button("analyze.measure", "Measure", "M"),
    ];
    ITEMS
}

pub(super) fn toolbar_icon_asset(icon: ToolbarIcon) -> (&'static str, &'static [u8]) {
    match icon {
        ToolbarIcon::Rect => ("rect", include_bytes!("assets/tools/rect.png").as_slice()),
        ToolbarIcon::Oval => ("oval", include_bytes!("assets/tools/oval.png").as_slice()),
        ToolbarIcon::Poly => ("poly", include_bytes!("assets/tools/poly.png").as_slice()),
        ToolbarIcon::Free => ("free", include_bytes!("assets/tools/free.png").as_slice()),
        ToolbarIcon::Line => ("line", include_bytes!("assets/tools/line.png").as_slice()),
        ToolbarIcon::Angle => ("angle", include_bytes!("assets/tools/line.png").as_slice()),
        ToolbarIcon::Point => ("point", include_bytes!("assets/tools/point.png").as_slice()),
        ToolbarIcon::Wand => ("wand", include_bytes!("assets/tools/wand.png").as_slice()),
        ToolbarIcon::Text => ("text", include_bytes!("assets/tools/text.png").as_slice()),
        ToolbarIcon::Open => ("open", include_bytes!("assets/tools/open.png").as_slice()),
        ToolbarIcon::Zoom => ("zoom", include_bytes!("assets/tools/zoom.png").as_slice()),
        ToolbarIcon::Hand => ("hand", include_bytes!("assets/tools/hand.png").as_slice()),
        ToolbarIcon::Dropper => (
            "dropper",
            include_bytes!("assets/tools/dropper.png").as_slice(),
        ),
        ToolbarIcon::Previous => (
            "previous",
            include_bytes!("assets/tools/previous.png").as_slice(),
        ),
        ToolbarIcon::Next => ("next", include_bytes!("assets/tools/next.png").as_slice()),
        ToolbarIcon::Quit => ("quit", include_bytes!("assets/tools/quit.png").as_slice()),
        ToolbarIcon::Custom1 => (
            "custom1",
            include_bytes!("assets/tools/custom1.png").as_slice(),
        ),
        ToolbarIcon::Custom2 => (
            "custom2",
            include_bytes!("assets/tools/custom2.png").as_slice(),
        ),
        ToolbarIcon::Custom3 => (
            "custom3",
            include_bytes!("assets/tools/custom3.png").as_slice(),
        ),
        ToolbarIcon::More => ("more", include_bytes!("assets/tools/more.png").as_slice()),
    }
}

pub(super) fn tool_from_command_id(command_id: &str) -> Option<ToolId> {
    interaction::tooling::tool_from_command_id(command_id)
}

pub(super) fn tool_shortcut_command(text: &str) -> Option<&'static str> {
    interaction::tooling::tool_shortcut_command(text)
}

#[allow(dead_code)]
pub(super) fn tool_shortcut_tool(text: &str) -> Option<ToolId> {
    interaction::tooling::tool_shortcut_tool(text)
}
